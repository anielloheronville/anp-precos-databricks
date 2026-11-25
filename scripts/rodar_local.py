"""Roda o pipeline inteiro na máquina local e gera reports/.

Uso:
    python scripts/rodar_local.py --origem data/raw/csv --lago data/lago --nucleos 6
    python scripts/rodar_local.py --etapas repasse      # só barril, bomba, cenários e eventos (rápido)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
# workers do Spark no mesmo Python do script (no Windows, "python3" é o atalho da Microsoft Store)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from pyspark.sql import SparkSession  # noqa: E402

from anp_precos import estatistica, eventos, graficos, modelo, pipeline, repasse  # noqa: E402
from anp_precos.esquema import ARQUIVOS_REFERENCIA, CORTE_PARIDADE, EVENTOS  # noqa: E402
from anp_precos.lago import Lago  # noqa: E402


def sessao(nucleos: int, memoria: str) -> SparkSession:
    return (
        SparkSession.builder.master(f"local[{nucleos}]")
        .appName("anp-precos")
        .config("spark.driver.memory", memoria)
        .config("spark.sql.shuffle.partitions", str(nucleos * 4))
        .config("spark.sql.session.timeZone", "America/Sao_Paulo")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def _limpo(valor):
    """NaN vira null (JSON não tem NaN) e -0.0 vira 0.0."""
    if isinstance(valor, dict):
        return {chave: _limpo(v) for chave, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_limpo(v) for v in valor]
    if isinstance(valor, float):
        return None if valor != valor else valor + 0.0
    return valor


def gravar_json(caminho: Path, conteudo: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(_limpo(conteudo), indent=2, ensure_ascii=False, allow_nan=False)
    caminho.write_text(texto + "\n", "utf-8")


def etapa_precos(
    spark: SparkSession, lago: Lago, origem: Path, relatorio: Path, resumo_ref: dict, tempos: dict
) -> None:
    """Coletas da ANP: bronze, silver, gold, prêmio da Ipiranga e modelo de alta."""
    figuras = relatorio / "figuras"
    # lista de arquivos em vez da pasta: no Windows, listar pasta pelo Hadoop exige winutils.exe
    arquivos = sorted(str(p) for p in origem.glob("*.csv"))
    if not arquivos:
        raise SystemExit(f"Nenhum CSV em {origem}")
    t = time.perf_counter()
    pipeline.rodar_bronze(spark, lago, arquivos)
    tempos["bronze_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    qualidade = pipeline.rodar_silver(lago)
    tempos["silver_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    cobertura = pipeline.rodar_gold(lago)
    tempos["gold_s"] = round(time.perf_counter() - t, 1)
    gravar_json(relatorio / "qualidade.json", qualidade)
    print("qualidade:", json.dumps(qualidade, ensure_ascii=False))

    mercado = lago.ler("gold_mercado_semanal").toPandas()
    por_bandeira = lago.ler("gold_bandeira_semanal").toPandas()
    feats = lago.ler("gold_features").toPandas()

    premio = estatistica.premio_por_produto(mercado)
    premio.to_csv(relatorio / "premio_ipiranga_por_produto.csv", index=False)
    print(premio.to_string(index=False))

    t = time.perf_counter()
    resultado = modelo.treinar_e_avaliar(feats)
    importancias = modelo.importancia_por_permutacao(resultado)
    comparacao = modelo.comparar_sem_brent(feats, resultado)
    tempos["modelo_s"] = round(time.perf_counter() - t, 1)

    metricas = {k: v for k, v in resultado.items() if not k.startswith("_")}
    metricas["importancia_permutacao"] = importancias
    metricas["efeito_do_brent"] = comparacao
    metricas["referencias"] = resumo_ref
    metricas["cobertura_do_brent"] = cobertura
    metricas["tempos"] = tempos
    metricas["linhas_gold"] = {"mercado_semanal": len(mercado), "features": len(feats)}
    gravar_json(relatorio / "metricas_modelo.json", metricas)
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    for produto in ["GASOLINA", "DIESEL S10", "ETANOL"]:
        graficos.evolucao_por_bandeira(por_bandeira, produto, figuras)
        graficos.premio_por_uf(mercado, produto, figuras)
    graficos.qualidade(qualidade, figuras)
    graficos.curva_roc(resultado, figuras)
    graficos.importancia(importancias, figuras)


def etapa_repasse(spark: SparkSession, lago: Lago, pasta_ref: Path, relatorio: Path) -> None:
    """Barril contra bomba, cenários e estudo de eventos, sobre as séries de referência."""
    figuras = relatorio / "figuras"
    caminhos = {serie: str(pasta_ref / arquivo) for serie, arquivo in ARQUIVOS_REFERENCIA.items()}
    resumo_macro = pipeline.rodar_macro(spark, lago, caminhos)
    print("bomba e séries mensais:", json.dumps(resumo_macro, ensure_ascii=False))
    semanal = lago.ler("gold_repasse_semanal").toPandas()
    mensal = lago.ler("gold_macro_mensal").toPandas()

    analise = repasse.relatorio(semanal)
    analise["cenarios"].to_csv(relatorio / "cenarios_bomba.csv", index=False)
    tabela_eventos = eventos.tabela(mensal)
    tabela_eventos.to_csv(relatorio / "eventos.csv", index=False)
    correlacoes = eventos.correlacoes(mensal)
    gravar_json(
        relatorio / "repasse_e_eventos.json",
        {
            "dados": resumo_macro,
            "situacao_atual": analise["situacao_atual"],
            "hipoteses": analise["hipoteses"],
            "repasse": repasse.resumo_das_curvas(analise["curvas"]),
            "alimentos": {
                "correlacao_com_variacao_k_meses_antes": {
                    serie: {str(k): v for k, v in linha.items()} for serie, linha in correlacoes.iterrows()
                },
                "canais": eventos.canais(mensal),
            },
        },
    )
    graficos.barril_e_bomba(semanal, figuras, corte=CORTE_PARIDADE)
    graficos.curvas_de_repasse(analise["curvas"], figuras)
    graficos.cenarios(analise["cenarios"], figuras)
    graficos.eventos_e_alimentos(eventos.preparar(mensal), EVENTOS, figuras)
    print(analise["cenarios"][analise["cenarios"]["semanas_depois"] == 16].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default=str(RAIZ / "data" / "raw" / "csv"))
    ap.add_argument("--referencias", default=str(RAIZ / "referencias"), help="pasta com as séries de referência")
    ap.add_argument("--lago", default=str(RAIZ / "data" / "lago"))
    ap.add_argument("--relatorio", default=str(RAIZ / "reports"))
    ap.add_argument("--nucleos", type=int, default=6)
    ap.add_argument("--memoria", default="8g")
    ap.add_argument("--etapas", choices=["tudo", "repasse"], default="tudo", help="repasse = sem coletas e modelo")
    ap.add_argument(
        "--modo",
        choices=["memoria", "parquet"],
        default="memoria" if os.name == "nt" else "parquet",
        help="memoria = sem gravar pelo Spark (Windows sem winutils.exe); parquet = grava em --lago",
    )
    args = ap.parse_args()

    spark = sessao(args.nucleos, args.memoria)
    spark.sparkContext.setLogLevel("ERROR")
    lago = Lago(spark, em_memoria=True) if args.modo == "memoria" else Lago(spark, pasta_local=args.lago)
    relatorio, pasta_ref = Path(args.relatorio), Path(args.referencias)
    relatorio.mkdir(parents=True, exist_ok=True)
    tempos: dict[str, float] = {}

    # as referências vêm primeiro: são rápidas, e arquivo ruim para tudo antes do trabalho pesado
    t = time.perf_counter()
    resumo_ref = pipeline.rodar_referencias(
        spark, lago, str(pasta_ref / ARQUIVOS_REFERENCIA["brent"]), str(pasta_ref / ARQUIVOS_REFERENCIA["dolar"])
    )
    tempos["referencias_s"] = round(time.perf_counter() - t, 1)
    print("referências:", json.dumps(resumo_ref, ensure_ascii=False))

    if args.etapas == "tudo":
        etapa_precos(spark, lago, Path(args.origem), relatorio, resumo_ref, tempos)
    etapa_repasse(spark, lago, pasta_ref, relatorio)
    spark.stop()


if __name__ == "__main__":
    main()
