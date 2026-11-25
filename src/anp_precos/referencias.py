"""Séries de referência do mercado: Brent e dólar, para chegar ao barril em reais.

- Bronze: o CSV de cada fonte como veio, tudo texto, com o arquivo de origem.
- Silver: uma linha por dia, tipada (`brent_usd`, `dolar`).
- Gold: uma linha por semana, com o barril em reais por litro e suas variações.

A semana é a mesma do resto do projeto (segunda-feira), para casar com as coletas da ANP.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from anp_precos.esquema import (
    COLUNAS_BOMBA,
    COLUNAS_BRENT,
    FAIXA_PRECO_BOMBA,
    FONTES_REFERENCIA,
    LITROS_POR_BARRIL,
    MIN_DIAS_SEMANA,
    PASSOS_BRENT,
    PRODUTOS_BOMBA,
    PRODUTOS_REPASSE,
)

BOM = chr(0xFEFF)  # por código, para nenhum editor trocar por um caractere invisível

__all__ = [
    "COLUNAS_BRENT",
    "bomba_invalidas",
    "bomba_semanal",
    "brent_semanal",
    "diario",
    "invalidas",
    "ler",
    "ler_bomba",
    "macro_mensal",
    "mensal",
    "repasse_semanal",
]


def _ler_csv(spark: SparkSession, caminho: str | list[str], sep: str, colunas: dict[str, str], fonte: str) -> DataFrame:
    """CSV como veio, só com as colunas esperadas (renomeadas), mais o arquivo de origem.

    FAILFAST: linha com colunas a mais ou a menos (arquivo corrompido, separador
    trocado) derruba a leitura, em vez de ser encaixada à força no esquema.
    """
    bruto = (
        spark.read.option("header", True)
        .option("sep", sep)
        .option("encoding", "UTF-8")
        .option("quote", '"')
        .option("inferSchema", False)
        .option("enforceSchema", False)
        .option("mode", "FAILFAST")
        .csv(caminho)
    )
    limpas = {coluna: coluna.replace(BOM, "").strip() for coluna in bruto.columns}
    faltando = sorted(set(colunas) - set(limpas.values()))
    if faltando:
        raise ValueError(f"Cabeçalho da série '{fonte}' mudou; faltam colunas: {faltando}")
    bronze = bruto.select(
        [F.col(f"`{original}`").alias(colunas[limpa]) for original, limpa in limpas.items() if limpa in colunas]
    )
    return bronze.withColumn("_arquivo", F.col("_metadata.file_name")).withColumn(
        "_ingestao_em", F.current_timestamp()
    )


def ler(spark: SparkSession, caminho: str | list[str], fonte: str) -> DataFrame:
    """Lê o CSV de uma série de data e valor (ver FONTES_REFERENCIA) sem converter nada."""
    formato = FONTES_REFERENCIA[fonte]
    return _ler_csv(spark, caminho, formato["sep"], formato["colunas"], fonte)


def _tipar(bronze: DataFrame, fonte: str) -> DataFrame:
    """Converte data e valor com try_* (modo ANSI): o que não converte vira nulo, não erro."""
    formato = FONTES_REFERENCIA[fonte]["formato_data"]
    return bronze.select(
        F.col("data").alias("data_texto"),
        F.col("valor").alias("valor_texto"),
        F.to_date(F.expr(f"try_to_timestamp(trim(data), '{formato}')")).alias("data"),
        F.expr("try_cast(regexp_replace(trim(valor), ',', '.') as double)").alias("valor"),
    )


def _sem_cotacao() -> F.Column:
    """Dia sem cotação (feriado): a fonte manda o valor vazio ou "."."""
    return F.trim(F.coalesce(F.col("valor_texto"), F.lit(""))).isin("", ".")


def _na_faixa(fonte: str) -> F.Column:
    """Valor plausível para a série. NaN e infinito ficam de fora: no Spark, NaN é maior que tudo."""
    minimo, maximo = FONTES_REFERENCIA[fonte]["faixa"]
    return F.col("valor").isNotNull() & ~F.isnan("valor") & F.col("valor").between(minimo, maximo)


def invalidas(bronze: DataFrame, fonte: str) -> DataFrame:
    """Linhas que a fonte preencheu e o pipeline não entendeu.

    Dia sem cotação não é erro. Data ilegível, texto que não vira número, NaN,
    infinito ou valor fora da faixa plausível da série são: quem chama deve
    falhar alto em vez de seguir com um buraco silencioso.
    """
    tipado = _tipar(bronze, fonte)
    valor_ruim = ~_sem_cotacao() & ~_na_faixa(fonte)
    return tipado.where(F.col("data").isNull() | valor_ruim).select(
        F.lit(fonte).alias("fonte"), "data_texto", "valor_texto"
    )


def diario(bronze_brent: DataFrame, bronze_dolar: DataFrame) -> DataFrame:
    """Uma linha por dia com `brent_usd` (US$/barril) e `dolar` (R$/US$).

    Cada fonte tem o seu calendário de feriados, então a junção é externa: um
    dia pode ter só uma das duas cotações. Se o mesmo dia aparecer mais de uma
    vez (downloads sobrepostos na pasta), conta uma vez só, pela média.
    """

    def serie(bronze: DataFrame, fonte: str, nome: str) -> DataFrame:
        validos = _tipar(bronze, fonte).where(F.col("data").isNotNull() & _na_faixa(fonte))
        return validos.groupBy("data").agg(F.avg("valor").alias(nome))

    return serie(bronze_brent, "brent", "brent_usd").join(
        serie(bronze_dolar, "dolar", "dolar"), on="data", how="full"
    )


def brent_semanal(diario: DataFrame) -> DataFrame:
    """Uma linha por semana: médias da semana, barril em reais por litro e variações.

    O barril em reais é a média do Brent vezes a média do dólar na semana, e só
    existe quando as duas séries têm pelo menos MIN_DIAS_SEMANA cotações. As
    variações comparam com a semana exatamente N semanas antes: se ela não
    existe ou está incompleta, a variação fica nula, nunca emprestada de outra.
    """
    semanal = (
        diario.withColumn("semana", F.to_date(F.date_trunc("week", F.col("data"))))
        .groupBy("semana")
        .agg(
            F.avg("brent_usd").alias("brent_usd"),
            F.count("brent_usd").alias("dias_brent"),
            F.avg("dolar").alias("dolar"),
            F.count("dolar").alias("dias_dolar"),
        )
    )
    completa = (F.col("dias_brent") >= MIN_DIAS_SEMANA) & (F.col("dias_dolar") >= MIN_DIAS_SEMANA)
    semanal = semanal.withColumn(
        "brent_reais_litro", F.when(completa, F.col("brent_usd") * F.col("dolar") / LITROS_POR_BARRIL)
    )
    precos = semanal.select("semana", "brent_reais_litro")
    for passos in PASSOS_BRENT:
        anterior = precos.select(
            F.date_add("semana", 7 * passos).alias("semana"),
            F.col("brent_reais_litro").alias("_anterior"),
        )
        semanal = (
            semanal.join(anterior, on="semana", how="left")
            .withColumn(f"var_brent_{passos}s", F.col("brent_reais_litro") / F.col("_anterior") - 1)
            .drop("_anterior")
        )
    return semanal


# ---------------------------------------------------------------------------------------------
# Bomba (ANP, Brasil, semanal) e séries mensais: base do repasse, dos cenários e do estudo de eventos
# ---------------------------------------------------------------------------------------------


def ler_bomba(spark: SparkSession, caminho: str | list[str]) -> DataFrame:
    """Lê o CSV do preço médio semanal de revenda no Brasil (planilha da ANP já convertida)."""
    return _ler_csv(spark, caminho, ",", {coluna: coluna for coluna in COLUNAS_BOMBA}, "bomba_brasil")


def _tipar_bomba(bronze: DataFrame) -> DataFrame:
    mapa = F.create_map([F.lit(texto) for par in PRODUTOS_BOMBA.items() for texto in par])
    minimo, maximo = FAIXA_PRECO_BOMBA
    preco = F.expr("try_cast(trim(preco_medio_revenda) as double)")
    return bronze.select(
        "data_inicial",
        F.col("produto").alias("produto_texto"),
        F.col("preco_medio_revenda").alias("preco_texto"),
        F.to_date(F.expr("try_to_timestamp(trim(data_inicial), 'yyyy-MM-dd')")).alias("inicio"),
        F.try_element_at(mapa, F.upper(F.trim(F.col("produto")))).alias("produto"),
        preco.alias("preco_medio"),
        F.expr("try_cast(try_cast(trim(numero_de_postos) as double) as int)").alias("n_postos"),
    ).withColumn(
        "_valida",
        # a semana da ANP vai de domingo a sábado; qualquer outra coisa é arquivo errado
        F.col("inicio").isNotNull()
        & (F.dayofweek("inicio") == 1)
        & F.col("produto").isNotNull()
        & F.col("preco_medio").isNotNull()
        & ~F.isnan("preco_medio")
        & F.col("preco_medio").between(minimo, maximo),
    )


def bomba_invalidas(bronze: DataFrame) -> DataFrame:
    """Linhas da série semanal que o pipeline não entendeu: data, produto ou preço fora do esperado."""
    return _tipar_bomba(bronze).where(~F.col("_valida")).select("data_inicial", "produto_texto", "preco_texto")


def bomba_semanal(bronze: DataFrame) -> DataFrame:
    """Uma linha por semana e produto, com a semana do projeto (segunda-feira).

    A ANP data a semana pelo domingo; as coletas acontecem de segunda a sexta,
    então a segunda-feira seguinte ao domingo é a mesma semana dos microdados.
    """
    return (
        _tipar_bomba(bronze)
        .where(F.col("_valida"))
        .select(F.date_add("inicio", 1).alias("semana"), "produto", "preco_medio", "n_postos")
        .groupBy("semana", "produto")
        .agg(F.avg("preco_medio").alias("preco_medio"), F.max("n_postos").alias("n_postos"))
    )


def _bomba_larga(bomba: DataFrame, chave: F.Column, nome: str) -> DataFrame:
    """Um produto por coluna (gasolina, diesel_s10, ...), agregado pela chave de tempo."""
    largo = (
        bomba.where(F.col("produto").isin(list(PRODUTOS_REPASSE)))
        .groupBy(chave.alias(nome))
        .pivot("produto", list(PRODUTOS_REPASSE))
        .agg(F.avg("preco_medio"))
    )
    for produto, coluna in PRODUTOS_REPASSE.items():
        largo = largo.withColumnRenamed(produto, coluna)
    return largo


def repasse_semanal(bomba: DataFrame, brent: DataFrame) -> DataFrame:
    """Tabela do repasse: por semana, o preço na bomba de cada produto ao lado do barril em reais."""
    return _bomba_larga(bomba, F.col("semana"), "semana").join(
        brent.select("semana", "brent_usd", "dias_brent", "dolar", "dias_dolar", "brent_reais_litro"),
        on="semana",
        how="full",
    )


def mensal(bronze: DataFrame, fonte: str) -> DataFrame:
    """Série mensal tipada (`mes`, `<fonte>`), só com valores válidos."""
    validos = _tipar(bronze, fonte).where(F.col("data").isNotNull() & _na_faixa(fonte))
    return validos.groupBy(F.trunc("data", "month").alias("mes")).agg(F.avg("valor").alias(fonte))


def macro_mensal(diario: DataFrame, bomba: DataFrame, series: dict[str, DataFrame]) -> DataFrame:
    """Uma linha por mês: dólar, barril, bomba, IPCA e alimentos no mundo.

    `series` são as tabelas de `mensal` (ipca_geral, ipca_alimentos, alimentos_mundo).
    O barril e os alimentos em reais usam o dólar médio do mês.
    """
    saida = (
        diario.groupBy(F.trunc("data", "month").alias("mes"))
        .agg(F.avg("brent_usd").alias("brent_usd"), F.avg("dolar").alias("dolar"))
        .withColumn("barril_reais", F.col("brent_usd") * F.col("dolar"))
        .join(_bomba_larga(bomba, F.trunc("semana", "month"), "mes"), on="mes", how="full")
    )
    for tabela in series.values():
        saida = saida.join(tabela, on="mes", how="full")
    if "alimentos_mundo" in series:
        saida = saida.withColumn("alimentos_mundo_reais", F.col("alimentos_mundo") * F.col("dolar"))
    return saida
