"""Silver: tipos corretos, regras de qualidade, quarentena e deduplicação."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from anp_precos.esquema import (
    APELIDOS_BANDEIRA,
    BANDEIRAS_PRINCIPAIS,
    COLUNAS_BRUTAS,
    PRECO_MAX,
    PRECO_MIN,
    PRODUTOS,
    REGIOES,
    UFS,
    UNIDADES,
)

COLUNAS_SILVER = [
    "regiao",
    "uf",
    "municipio",
    "cnpj",
    "bairro",
    "cep",
    "produto",
    "unidade",
    "data_coleta",
    "semana",
    "valor_venda",
    "bandeira",
    "bandeira_original",
    "_arquivo",
]


def _texto(coluna: str) -> F.Column:
    """Tira espaços nas pontas e troca texto vazio por nulo."""
    limpo = F.trim(F.col(coluna))
    return F.when(limpo == "", None).otherwise(limpo)


def tipar(bronze: DataFrame) -> DataFrame:
    """Converte os textos do bronze em tipos, sem nunca derrubar a leitura.

    Spark 4 e o serverless do Databricks rodam em modo ANSI: um cast inválido
    lança erro. Por isso todo parse usa try_*, e o que falhar vira nulo e cai
    na quarentena com o motivo.
    """
    df = bronze
    for coluna in ["regiao", "uf", "municipio", "bairro", "cep", "produto", "unidade", "bandeira"]:
        df = df.withColumn(coluna, F.upper(_texto(coluna)))
    # 2025-02 grava "R$ / m3" (dígito 3); os outros semestres, "R$ / m³"
    df = df.withColumn("unidade", F.regexp_replace(F.col("unidade"), "M(3|³)$", "m³"))
    df = df.withColumn("unidade", F.regexp_replace(F.col("unidade"), "LITRO", "litro"))
    # CNPJ alfanumérico existe desde jul/2026: mantém letras e dígitos, tira pontuação
    df = df.withColumn(
        "cnpj", F.regexp_replace(F.upper(F.coalesce(F.col("cnpj"), F.lit(""))), "[^0-9A-Z]", "")
    )
    df = df.withColumn(
        "valor_venda",
        F.expr("try_cast(regexp_replace(trim(valor_venda), ',', '.') as double)"),
    )
    df = df.withColumn(
        "data_coleta", F.to_date(F.expr("try_to_timestamp(trim(data_coleta), 'dd/MM/yyyy')"))
    )
    df = df.withColumn("semana", F.to_date(F.date_trunc("week", F.col("data_coleta"))))
    df = df.withColumn("bandeira_original", F.col("bandeira"))
    df = df.withColumn("bandeira", F.coalesce(_apelido(F.col("bandeira")), F.col("bandeira")))
    df = df.withColumn(
        "bandeira",
        F.when(F.col("bandeira").isin(BANDEIRAS_PRINCIPAIS), F.col("bandeira")).otherwise(
            F.lit("OUTRAS")
        ),
    )
    return df


def _apelido(coluna: F.Column) -> F.Column:
    expressao = F.lit(None).cast("string")
    for antigo, novo in APELIDOS_BANDEIRA.items():
        expressao = F.when(coluna == antigo, F.lit(novo)).otherwise(expressao)
    return expressao


def motivo_rejeicao() -> F.Column:
    """Primeira regra de qualidade que a linha quebra (nulo = linha válida)."""
    # linha ";;;;;;" (2025-01 tem ~9 mil): todas as colunas da ANP vazias
    vazia = F.lit(True)
    for coluna in COLUNAS_BRUTAS.values():
        texto = (F.col("bandeira_original") if coluna == "bandeira" else F.col(coluna)).cast("string")
        vazia = vazia & (texto.isNull() | (F.trim(texto) == ""))
    regras = [
        (vazia, "linha_vazia"),
        (~F.col("produto").isin(PRODUTOS) | F.col("produto").isNull(), "produto_invalido"),
        (~F.col("unidade").isin(UNIDADES) | F.col("unidade").isNull(), "unidade_invalida"),
        (~F.col("uf").isin(UFS) | F.col("uf").isNull(), "uf_invalida"),
        (~F.col("regiao").isin(REGIOES) | F.col("regiao").isNull(), "regiao_invalida"),
        (F.length("cnpj") != 14, "cnpj_invalido"),
        (F.col("data_coleta").isNull(), "data_invalida"),
        (F.col("valor_venda").isNull(), "preco_ausente"),
        (~F.col("valor_venda").between(PRECO_MIN, PRECO_MAX), "preco_fora_da_faixa"),
        (F.col("municipio").isNull(), "municipio_ausente"),
    ]
    expressao = F.lit(None).cast("string")
    for condicao, motivo in reversed(regras):
        expressao = F.when(condicao, F.lit(motivo)).otherwise(expressao)
    return expressao


def separar(bronze: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Devolve (silver válido e deduplicado, quarentena com motivo)."""
    tipado = tipar(bronze).withColumn("motivo_rejeicao", motivo_rejeicao())
    quarentena = tipado.where(F.col("motivo_rejeicao").isNotNull())
    validos = tipado.where(F.col("motivo_rejeicao").isNull())
    # A ANP às vezes repete a mesma coleta; fica uma linha por posto, produto e dia.
    silver = validos.dropDuplicates(["cnpj", "produto", "data_coleta"]).select(COLUNAS_SILVER)
    return silver, quarentena


def resumo_qualidade(bronze: DataFrame, silver: DataFrame, quarentena: DataFrame) -> dict:
    """Números para o relatório de qualidade: quanto entrou, quanto ficou e por quê."""
    total = bronze.count()
    validos = silver.count()
    motivos = {
        linha["motivo_rejeicao"]: linha["n"]
        for linha in quarentena.groupBy("motivo_rejeicao").agg(F.count("*").alias("n")).collect()
    }
    rejeitadas = sum(motivos.values())
    return {
        "linhas_bronze": total,
        "linhas_silver": validos,
        "linhas_quarentena": rejeitadas,
        "duplicatas_removidas": total - rejeitadas - validos,
        "motivos": dict(sorted(motivos.items(), key=lambda kv: -kv[1])),
    }
