"""Bronze: o CSV da ANP como veio, tudo texto, com a origem de cada linha."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from anp_precos.esquema import COLUNAS_BRUTAS


def ler_csv_anp(spark: SparkSession, caminho: str | list[str]) -> DataFrame:
    """Lê um arquivo, uma lista de arquivos ou uma pasta de CSVs semestrais da ANP.

    Os arquivos são UTF-8 com BOM, separados por ';', com vírgula decimal e alguns
    campos entre aspas com quebra de linha dentro (por isso multiLine). Nada é
    convertido aqui: tipos e validação ficam no silver.
    """
    bruto = (
        spark.read.option("header", True)
        .option("sep", ";")
        .option("encoding", "UTF-8")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", True)
        .option("inferSchema", False)
        # confere o cabeçalho de CADA arquivo contra o primeiro, não só o do primeiro
        .option("enforceSchema", False)
        .csv(caminho)
    )
    renomes = {}
    for coluna in bruto.columns:
        limpa = coluna.replace("\ufeff", "").strip()
        renomes[coluna] = COLUNAS_BRUTAS.get(limpa, limpa.lower().replace(" ", "_"))
    bronze = bruto.select([F.col(f"`{c}`").alias(n) for c, n in renomes.items()])
    faltando = sorted(set(COLUNAS_BRUTAS.values()) - set(bronze.columns))
    if faltando:
        raise ValueError(f"Cabeçalho da ANP mudou; faltam colunas: {faltando}")
    return bronze.withColumn("_arquivo", F.col("_metadata.file_name")).withColumn(
        "_ingestao_em", F.current_timestamp()
    )
