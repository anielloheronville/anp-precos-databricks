"""Orquestra bronze -> silver -> gold. Usado pelo script local e pelos notebooks."""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from anp_precos import bronze, gold, referencias, silver
from anp_precos.esquema import COLUNAS_BRENT, MAX_SEMANAS_SEM_BRENT
from anp_precos.lago import Lago

COMENTARIOS = {
    "bronze_brent": "Brent spot em US$/barril, diário, como veio (U.S. EIA, série DCOILBRENTEU do FRED).",
    "bronze_dolar": "Dólar PTAX de venda em R$/US$, diário, como veio (Banco Central, série 1 do SGS).",
    "silver_referencias_diarias": "Brent e dólar tipados, uma linha por dia.",
    "gold_brent_semanal": "Barril de Brent em reais por litro, por semana, com variações de 1, 4 e 8 semanas.",
    "bronze_ipca_geral": "IPCA, variação mensal em %, como veio (IBGE, série 433 do SGS do Banco Central).",
    "bronze_ipca_alimentos": "IPCA de alimentação e bebidas, % ao mês, como veio (IBGE, série 1635 do SGS).",
    "bronze_alimentos_mundo": "Índice mundial de alimentos em dólares, mensal, como veio (FMI, PFOODINDEXM do FRED).",
    "bronze_bomba_brasil": "Preço médio semanal de revenda no Brasil, como veio (ANP, planilha convertida em CSV).",
    "silver_bomba_brasil": "Preço médio de revenda no Brasil por semana e produto, tipado.",
    "gold_repasse_semanal": "Por semana: preço na bomba (gasolina, diesel, etanol) ao lado do barril em reais.",
    "gold_macro_mensal": "Por mês: dólar, barril, bomba, IPCA geral e de alimentos, e alimentos no mundo.",
    "bronze_precos": "Coletas semanais de preço da ANP como vieram (texto), com arquivo de origem.",
    "silver_precos": "Coletas válidas, tipadas e deduplicadas (1 por posto, produto e dia).",
    "silver_quarentena": "Linhas rejeitadas pelas regras de qualidade, com o motivo.",
    "gold_mercado_semanal": "Preço por UF, município, produto e semana, com prêmio Ipiranga vs. concorrentes.",
    "gold_bandeira_semanal": "Mediana nacional por produto, bandeira e semana.",
    "gold_nacional_semanal": "Mediana nacional por produto e semana, com variações.",
    "gold_features": "Features (preço local, nacional e barril em reais) e alvo (alta >= 0,5% na semana seguinte).",
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


def _falhar_se_houver(invalidas, fonte: str) -> None:
    """Série de referência com linha que o pipeline não entende derruba a carga, com exemplos."""
    ruins = invalidas.limit(5).collect()
    if ruins:
        exemplos = [tuple(r) for r in ruins]
        raise ValueError(f"Série '{fonte}' tem linhas que não são data e número na faixa esperada: {exemplos}")


def rodar_referencias(spark: SparkSession, lago: Lago, caminho_brent: str, caminho_dolar: str) -> dict:
    """Brent e dólar: bronze, silver diário e gold semanal. Precisa rodar antes do gold."""
    for fonte, caminho in [("brent", caminho_brent), ("dolar", caminho_dolar)]:
        tabela = f"bronze_{fonte}"
        lago.gravar(referencias.ler(spark, caminho, fonte), tabela, comentario=COMENTARIOS[tabela])
        _falhar_se_houver(referencias.invalidas(lago.ler(tabela), fonte), fonte)
    lago.gravar(
        referencias.diario(lago.ler("bronze_brent"), lago.ler("bronze_dolar")),
        "silver_referencias_diarias",
        comentario=COMENTARIOS["silver_referencias_diarias"],
    )
    lago.gravar(
        referencias.brent_semanal(lago.ler("silver_referencias_diarias")),
        "gold_brent_semanal",
        comentario=COMENTARIOS["gold_brent_semanal"],
    )
    com_preco = F.col("brent_reais_litro").isNotNull()
    resumo = (
        lago.ler("gold_brent_semanal")
        .agg(
            F.count("*").alias("semanas"),
            F.count("brent_reais_litro").alias("semanas_com_preco"),
            F.min(F.when(com_preco, F.col("semana"))).alias("primeira_semana_com_preco"),
            F.max(F.when(com_preco, F.col("semana"))).alias("ultima_semana_com_preco"),
        )
        .first()
    )
    if resumo["semanas_com_preco"] == 0:
        raise ValueError(
            "Nenhuma semana com barril em reais: os arquivos de Brent e dólar estão vazios ou não se sobrepõem."
        )
    dias = lago.ler("silver_referencias_diarias").agg(
        F.count("brent_usd").alias("dias_brent"), F.count("dolar").alias("dias_dolar")
    ).first()
    return {
        "dias_brent": dias["dias_brent"],
        "dias_dolar": dias["dias_dolar"],
        "semanas": resumo["semanas"],
        "semanas_com_preco": resumo["semanas_com_preco"],
        "primeira_semana_com_preco": str(resumo["primeira_semana_com_preco"]),
        "ultima_semana_com_preco": str(resumo["ultima_semana_com_preco"]),
    }


def rodar_gold(lago: Lago) -> dict:
    """Tabelas gold. Precisa de `rodar_silver` e de `rodar_referencias` antes.

    Devolve quantas semanas do `gold_features` ficaram sem o barril em reais e
    falha se passarem de MAX_SEMANAS_SEM_BRENT: série de referência
    desatualizada ou cortada não pode virar feature nula em silêncio.
    """
    try:  # .columns resolve a tabela agora, antes de recalcular as outras três tabelas gold
        df_brent = lago.ler("gold_brent_semanal")
        faltando = sorted(set(COLUNAS_BRENT) - set(df_brent.columns))
    except Exception as erro:
        raise RuntimeError("Tabela gold_brent_semanal ausente: rode pipeline.rodar_referencias antes.") from erro
    if faltando:
        raise RuntimeError(f"gold_brent_semanal sem as colunas {faltando}: rode pipeline.rodar_referencias de novo.")
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
        gold.features(
            lago.ler("gold_mercado_semanal"),
            lago.ler("gold_nacional_semanal"),
            brent=df_brent,
        ),
        "gold_features",
        comentario=COMENTARIOS["gold_features"],
    )
    cobertura = (
        lago.ler("gold_features")
        .groupBy("semana")
        .agg(F.max(F.col(COLUNAS_BRENT[0]).isNotNull().cast("int")).alias("tem_brent"))
        .agg(F.count("*").alias("semanas"), F.sum("tem_brent").alias("com_brent"))
        .first()
    )
    semanas, sem_brent = cobertura["semanas"], cobertura["semanas"] - (cobertura["com_brent"] or 0)
    if semanas and sem_brent / semanas > MAX_SEMANAS_SEM_BRENT:
        raise ValueError(
            f"O barril em reais falta em {sem_brent} de {semanas} semanas do gold_features: "
            "atualize referencias/ (scripts/baixar_referencias.py) e rode as referências de novo."
        )
    return {"semanas": semanas, "semanas_sem_brent": sem_brent}


SERIES_MENSAIS = ["ipca_geral", "ipca_alimentos", "alimentos_mundo"]


def rodar_macro(spark: SparkSession, lago: Lago, caminhos: dict[str, str]) -> dict:
    """Bomba semanal do Brasil, IPCA e alimentos no mundo: base do repasse e do estudo de eventos.

    `caminhos` mapeia cada série (ver ARQUIVOS_REFERENCIA) ao seu arquivo.
    Precisa de `rodar_referencias` antes, porque usa o Brent e o dólar.
    """
    try:
        df_diario, df_brent = lago.ler("silver_referencias_diarias"), lago.ler("gold_brent_semanal")
        _ = df_diario.columns, df_brent.columns  # resolve as tabelas agora
    except Exception as erro:
        raise RuntimeError("Tabelas do Brent e do dólar ausentes: rode pipeline.rodar_referencias antes.") from erro
    for fonte in SERIES_MENSAIS:
        tabela = f"bronze_{fonte}"
        lago.gravar(referencias.ler(spark, caminhos[fonte], fonte), tabela, comentario=COMENTARIOS[tabela])
        _falhar_se_houver(referencias.invalidas(lago.ler(tabela), fonte), fonte)
        if referencias.mensal(lago.ler(tabela), fonte).limit(1).count() == 0:
            raise ValueError(f"Série '{fonte}' sem nenhum mês válido: o arquivo está vazio.")
    lago.gravar(
        referencias.ler_bomba(spark, caminhos["bomba_brasil"]),
        "bronze_bomba_brasil",
        comentario=COMENTARIOS["bronze_bomba_brasil"],
    )
    _falhar_se_houver(referencias.bomba_invalidas(lago.ler("bronze_bomba_brasil")), "bomba_brasil")
    lago.gravar(
        referencias.bomba_semanal(lago.ler("bronze_bomba_brasil")),
        "silver_bomba_brasil",
        comentario=COMENTARIOS["silver_bomba_brasil"],
    )
    lago.gravar(
        referencias.repasse_semanal(lago.ler("silver_bomba_brasil"), df_brent),
        "gold_repasse_semanal",
        comentario=COMENTARIOS["gold_repasse_semanal"],
    )
    series = {fonte: referencias.mensal(lago.ler(f"bronze_{fonte}"), fonte) for fonte in SERIES_MENSAIS}
    lago.gravar(
        referencias.macro_mensal(df_diario, lago.ler("silver_bomba_brasil"), series),
        "gold_macro_mensal",
        comentario=COMENTARIOS["gold_macro_mensal"],
    )
    tem_bomba = F.col("gasolina").isNotNull()
    semanal = (
        lago.ler("gold_repasse_semanal")
        .agg(
            F.count("gasolina").alias("semanas_com_bomba"),
            F.count(F.when(tem_bomba, F.col("brent_reais_litro"))).alias("semanas_com_bomba_e_barril"),
            F.max(F.when(tem_bomba, F.col("semana"))).alias("ultima_semana_da_bomba"),
        )
        .first()
    )
    mensal = (
        lago.ler("gold_macro_mensal")
        .agg(
            F.count("ipca_alimentos").alias("meses_com_ipca_de_alimentos"),
            F.max(F.when(F.col("ipca_alimentos").isNotNull(), F.col("mes"))).alias("ultimo_mes_do_ipca"),
        )
        .first()
    )
    if semanal["semanas_com_bomba_e_barril"] == 0 or mensal["meses_com_ipca_de_alimentos"] == 0:
        raise ValueError("As séries de bomba, barril e IPCA não se sobrepõem: confira os arquivos em referencias/.")
    return {
        "semanas_com_bomba": semanal["semanas_com_bomba"],
        "semanas_com_bomba_e_barril": semanal["semanas_com_bomba_e_barril"],
        "ultima_semana_da_bomba": str(semanal["ultima_semana_da_bomba"]),
        "meses_com_ipca_de_alimentos": mensal["meses_com_ipca_de_alimentos"],
        "ultimo_mes_do_ipca": str(mensal["ultimo_mes_do_ipca"])[:7],
    }
