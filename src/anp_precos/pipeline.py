"""Orquestra bronze -> silver -> gold. Usado pelo script local e pelos notebooks."""

from __future__ import annotations

from pyspark.sql import SparkSession

from anp_precos import bronze, gold, silver
from anp_precos.lago import Lago

COMENTARIOS = {
    "bronze_precos": "Coletas semanais de preço da ANP como vieram (texto), com arquivo de origem.",
    "silver_precos": "Coletas válidas, tipadas e deduplicadas (1 por posto, produto e dia).",
    "silver_quarentena": "Linhas rejeitadas pelas regras de qualidade, com o motivo.",
    "gold_mercado_semanal": "Preço por UF, município, produto e semana, com prêmio Ipiranga vs. concorrentes.",
    "gold_bandeira_semanal": "Mediana nacional por produto, bandeira e semana.",
    "gold_nacional_semanal": "Mediana nacional por produto e semana, com variações.",
    "gold_features": "Features e alvo (alta >= 0,5% na semana seguinte) por município, produto e semana.",
}


def rodar_bronze(spark: SparkSession, lago: Lago, origem: str | list[str]) -> None:
    lago.gravar(bronze.ler_csv_anp(spark, origem), "bronze_precos", comentario=COMENTARIOS["bronze_precos"])


def rodar_silver(lago: Lago) -> dict:
    df_bronze = lago.ler("bronze_precos")
    df_silver, df_quarentena = silver.separar(df_bronze)
    lago.gravar(
        df_silver,
        "silver_precos",
        comentario=COMENTARIOS["silver_precos"],
        cluster_por=["produto", "semana"],
    )
    lago.gravar(df_quarentena, "silver_quarentena", comentario=COMENTARIOS["silver_quarentena"])
    return silver.resumo_qualidade(df_bronze, lago.ler("silver_precos"), lago.ler("silver_quarentena"))


def rodar_gold(lago: Lago) -> None:
    df_silver = lago.ler("silver_precos")
    lago.gravar(
        gold.mercado_semanal(df_silver),
        "gold_mercado_semanal",
        comentario=COMENTARIOS["gold_mercado_semanal"],
        cluster_por=["produto", "semana"],
    )
    lago.gravar(
        gold.bandeira_semanal(df_silver), "gold_bandeira_semanal", comentario=COMENTARIOS["gold_bandeira_semanal"]
    )
    lago.gravar(
        gold.nacional_semanal(df_silver), "gold_nacional_semanal", comentario=COMENTARIOS["gold_nacional_semanal"]
    )
    lago.gravar(
        gold.features(lago.ler("gold_mercado_semanal"), lago.ler("gold_nacional_semanal")),
        "gold_features",
        comentario=COMENTARIOS["gold_features"],
    )
