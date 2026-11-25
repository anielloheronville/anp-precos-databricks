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
    COLUNAS_BRENT,
    FONTES_REFERENCIA,
    LITROS_POR_BARRIL,
    MIN_DIAS_SEMANA,
    PASSOS_BRENT,
)

BOM = chr(0xFEFF)  # por código, para nenhum editor trocar por um caractere invisível

__all__ = ["COLUNAS_BRENT", "brent_semanal", "diario", "invalidas", "ler"]


def ler(spark: SparkSession, caminho: str | list[str], fonte: str) -> DataFrame:
    """Lê o CSV de uma fonte (`brent` ou `dolar`) sem converter nada.

    FAILFAST: linha com colunas a mais ou a menos (arquivo corrompido, separador
    trocado) derruba a leitura, em vez de ser encaixada à força nas duas colunas.
    """
    formato = FONTES_REFERENCIA[fonte]
    bruto = (
        spark.read.option("header", True)
        .option("sep", formato["sep"])
        .option("encoding", "UTF-8")
        .option("quote", '"')
        .option("inferSchema", False)
        .option("enforceSchema", False)
        .option("mode", "FAILFAST")
        .csv(caminho)
    )
    limpas = {coluna: coluna.replace(BOM, "").strip() for coluna in bruto.columns}
    faltando = sorted(set(formato["colunas"]) - set(limpas.values()))
    if faltando:
        raise ValueError(f"Cabeçalho da série '{fonte}' mudou; faltam colunas: {faltando}")
    bronze = bruto.select(
        [F.col(f"`{original}`").alias(formato["colunas"][limpa]) for original, limpa in limpas.items()
         if limpa in formato["colunas"]]
    )
    return bronze.withColumn("_arquivo", F.col("_metadata.file_name")).withColumn(
        "_ingestao_em", F.current_timestamp()
    )


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
