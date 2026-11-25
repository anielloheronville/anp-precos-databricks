"""Roda o pipeline inteiro na máquina local e gera reports/.

Uso:
    python scripts/rodar_local.py --origem data/raw/csv --lago data/lago --nucleos 6
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

from pyspark.sql import SparkSession  # noqa: E402

from anp_precos import estatistica, graficos, modelo, pipeline  # noqa: E402
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default=str(RAIZ / "data" / "raw" / "csv"))
    ap.add_argument("--lago", default=str(RAIZ / "data" / "lago"))
    ap.add_argument("--relatorio", default=str(RAIZ / "reports"))
    ap.add_argument("--nucleos", type=int, default=6)
    ap.add_argument("--memoria", default="8g")
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
    relatorio = Path(args.relatorio)
    figuras = relatorio / "figuras"
    tempos: dict[str, float] = {}

    # lista de arquivos em vez da pasta: no Windows, listar pasta pelo Hadoop exige winutils.exe
    arquivos = sorted(str(p) for p in Path(args.origem).glob("*.csv"))
    if not arquivos:
        raise SystemExit(f"Nenhum CSV em {args.origem}")
    t = time.perf_counter()
    pipeline.rodar_bronze(spark, lago, arquivos)
    tempos["bronze_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    qualidade = pipeline.rodar_silver(lago)
    tempos["silver_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    pipeline.rodar_gold(lago)
    tempos["gold_s"] = round(time.perf_counter() - t, 1)
    relatorio.mkdir(parents=True, exist_ok=True)
    (relatorio / "qualidade.json").write_text(json.dumps(qualidade, indent=2, ensure_ascii=False), "utf-8")
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
    tempos["modelo_s"] = round(time.perf_counter() - t, 1)

    metricas = {k: v for k, v in resultado.items() if not k.startswith("_")}
    metricas["importancia_permutacao"] = importancias
    metricas["tempos"] = tempos
    metricas["linhas_gold"] = {"mercado_semanal": len(mercado), "features": len(feats)}
    (relatorio / "metricas_modelo.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    for produto in ["GASOLINA", "DIESEL S10", "ETANOL"]:
        graficos.evolucao_por_bandeira(por_bandeira, produto, figuras)
        graficos.premio_por_uf(mercado, produto, figuras)
    graficos.qualidade(qualidade, figuras)
    graficos.curva_roc(resultado, figuras)
    graficos.importancia(importancias, figuras)
    spark.stop()


if __name__ == "__main__":
    main()
