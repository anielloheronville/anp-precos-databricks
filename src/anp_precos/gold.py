"""Gold: mercado semanal por município, prêmio da Ipiranga e tabela de features."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from anp_precos.esquema import BANDEIRA_FOCO, COLUNAS_BRENT, DEFASAGEM_BRENT_SEMANAS, LIMIAR_ALTA

PRECISAO = 10000  # precisão do percentile_approx usado nos quartis


def _mediana(coluna: F.Column) -> F.Column:
    """Mediana exata (média dos dois do meio quando o número é par).

    percentile_approx(x, 0.5) devolveria a mediana INFERIOR: com 2 postos
    Ipiranga a 6,00 e 6,20, daria 6,00 e enviesaria o prêmio para baixo. Os
    grupos município x semana são pequenos, então a mediana exata é barata.
    """
    return F.median(coluna)


def mercado_semanal(silver: DataFrame, min_outros: int = 2) -> DataFrame:
    """Uma linha por UF, município, produto e semana.

    O prêmio da Ipiranga compara a mediana dos postos Ipiranga com a mediana dos
    demais postos do MESMO município, produto e semana. Só é calculado quando há
    pelo menos 1 posto Ipiranga e `min_outros` concorrentes, para não comparar
    um posto com ele mesmo.
    """
    eh_foco = F.col("bandeira") == BANDEIRA_FOCO
    preco = F.col("valor_venda")
    agregado = silver.groupBy("regiao", "uf", "municipio", "produto", "semana").agg(
        F.countDistinct("cnpj").alias("n_postos"),
        F.countDistinct(F.when(eh_foco, F.col("cnpj"))).alias("n_ipiranga"),
        F.countDistinct(F.when(~eh_foco, F.col("cnpj"))).alias("n_outros"),
        _mediana(preco).alias("preco_mediano"),
        F.percentile_approx(preco, 0.25, PRECISAO).alias("preco_p25"),
        F.percentile_approx(preco, 0.75, PRECISAO).alias("preco_p75"),
        F.avg(preco).alias("preco_medio"),
        _mediana(F.when(eh_foco, preco)).alias("preco_mediano_ipiranga"),
        _mediana(F.when(~eh_foco, preco)).alias("preco_mediano_outros"),
    )
    tem_comparacao = (F.col("n_ipiranga") >= 1) & (F.col("n_outros") >= min_outros)
    return agregado.withColumn(
        "dispersao", (F.col("preco_p75") - F.col("preco_p25")) / F.col("preco_mediano")
    ).withColumn(
        "premio_ipiranga_pct",
        F.when(
            tem_comparacao,
            (F.col("preco_mediano_ipiranga") / F.col("preco_mediano_outros") - 1) * 100,
        ),
    )


def bandeira_semanal(silver: DataFrame) -> DataFrame:
    """Mediana nacional por produto, bandeira e semana."""
    return silver.groupBy("produto", "bandeira", "semana").agg(
        _mediana(F.col("valor_venda")).alias("preco_mediano"),
        F.countDistinct("cnpj").alias("n_postos"),
    )


def nacional_semanal(silver: DataFrame) -> DataFrame:
    """Mediana nacional por produto e semana, com variações de 1 e 4 semanas."""
    w = Window.partitionBy("produto").orderBy("semana")
    nacional = silver.groupBy("produto", "semana").agg(
        _mediana(F.col("valor_venda")).alias("preco_nacional")
    )
    return (
        nacional.withColumn("_s1", F.lag("semana", 1).over(w))
        .withColumn("_s4", F.lag("semana", 4).over(w))
        .withColumn(
            "var_nacional_1s",
            F.when(
                F.datediff("semana", "_s1") == 7,
                F.col("preco_nacional") / F.lag("preco_nacional", 1).over(w) - 1,
            ),
        )
        .withColumn(
            "var_nacional_4s",
            F.when(
                F.datediff("semana", "_s4") == 28,
                F.col("preco_nacional") / F.lag("preco_nacional", 4).over(w) - 1,
            ),
        )
        .drop("_s1", "_s4")
    )


def _var(coluna: str, passos: int, w: Window) -> F.Column:
    """Variação contra `passos` semanas atrás, só se a semana anterior existir de fato.

    A ANP não visita todo município toda semana; sem esse cuidado, um "lag 1"
    poderia comparar com um mês atrás.
    """
    semana_ant = F.lag("semana", passos).over(w)
    return F.when(
        F.datediff(F.col("semana"), semana_ant) == 7 * passos,
        F.col(coluna) / F.lag(coluna, passos).over(w) - 1,
    )


def features(
    mercado: DataFrame,
    nacional: DataFrame,
    limiar: float = LIMIAR_ALTA,
    brent: DataFrame | None = None,
) -> DataFrame:
    """Tabela do modelo: uma linha por município, produto e semana.

    Alvo `sobe_prox_semana`: a mediana do município sobe pelo menos `limiar`
    na semana SEGUINTE. Linhas sem a semana seguinte observada ficam sem alvo
    (nulo) e saem do treino, mas continuam na tabela para pontuação.

    `brent` é a tabela semanal do barril em reais. A linha da semana t recebe
    as variações medidas até t-1 (DEFASAGEM_BRENT_SEMANAS): a fonte publica o
    Brent com cerca de uma semana de atraso, então é isso que existiria na hora
    de prever t+1. Sem `brent`, as colunas existem e ficam nulas.
    """
    w = Window.partitionBy("uf", "municipio", "produto").orderBy("semana")
    prox_semana = F.lead("semana", 1).over(w)
    prox_preco = F.lead("preco_mediano", 1).over(w)
    base = (
        mercado.withColumn("var_1s", _var("preco_mediano", 1, w))
        .withColumn("var_2s", _var("preco_mediano", 2, w))
        .withColumn("var_4s", _var("preco_mediano", 4, w))
        .withColumn(
            "sobe_prox_semana",
            F.when(
                F.datediff(prox_semana, F.col("semana")) == 7,
                ((prox_preco / F.col("preco_mediano") - 1) >= limiar).cast("int"),
            ),
        )
        .withColumn("semana_do_ano", F.weekofyear("semana"))
    )
    saida = base.join(
        nacional.select("produto", "semana", "var_nacional_1s", "var_nacional_4s"),
        on=["produto", "semana"],
        how="left",
    )
    colunas = [*saida.columns, *COLUNAS_BRENT]  # mesma ordem de colunas com ou sem a tabela do Brent
    if brent is None:
        for coluna in COLUNAS_BRENT:
            saida = saida.withColumn(coluna, F.lit(None).cast("double"))
        return saida.select(colunas)
    defasado = brent.select(
        F.date_add("semana", 7 * DEFASAGEM_BRENT_SEMANAS).alias("semana"), *COLUNAS_BRENT
    )
    return saida.join(defasado, on="semana", how="left").select(colunas)
