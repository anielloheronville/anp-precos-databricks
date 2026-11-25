# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze → Silver → Gold
# MAGIC
# MAGIC O mesmo código de `src/anp_precos` que roda (e é testado) localmente, aqui gravando
# MAGIC tabelas Delta no Unity Catalog, com comentário em cada tabela e liquid clustering
# MAGIC por produto e semana nas tabelas grandes. Além das coletas da ANP, entram o Brent e o
# MAGIC dólar, que viram o barril em reais por semana e alimentam as features do modelo.

# COMMAND ----------

import json
import os
import shutil
import sys
from pathlib import Path

RAIZ = Path(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, str(RAIZ / "src"))

from anp_precos import pipeline  # noqa: E402
from anp_precos.lago import Lago  # noqa: E402

dbutils.widgets.text("catalogo", "workspace")
dbutils.widgets.text("esquema", "anp")
catalogo = dbutils.widgets.get("catalogo")
esquema = dbutils.widgets.get("esquema")
volume = f"/Volumes/{catalogo}/{esquema}/bruto"
origem = f"{volume}/csv/"
lago = Lago(spark, catalogo=catalogo, esquema=esquema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Brent e dólar: o barril em reais
# MAGIC Séries diárias de fontes oficiais (U.S. EIA pelo FRED e Banco Central), versionadas em
# MAGIC `referencias/` no repositório. A célula copia os dois CSVs do Git folder para o volume e lê de lá
# MAGIC com o Spark, como os arquivos da ANP. Vem primeiro porque é rápida: se faltar arquivo ou houver
# MAGIC linha inválida, o notebook para aqui, antes do trabalho pesado.
# MAGIC
# MAGIC A semana só ganha preço com pelo menos 3 cotações de cada série, e a variação só existe contra a
# MAGIC semana exata.

# COMMAND ----------

arquivos_ref = ["brent_usd_diario.csv", "dolar_ptax_venda_diario.csv"]
destino_ref = Path(volume) / "referencias"
destino_ref.mkdir(exist_ok=True)
for nome in arquivos_ref:
    fonte = RAIZ / "referencias" / nome
    if not fonte.exists():
        raise FileNotFoundError(f"{fonte} não existe: faça pull do Git folder na branch que tem a pasta referencias/.")
    shutil.copyfile(fonte, destino_ref / nome)
    print(f"{nome} -> {destino_ref} ({fonte.stat().st_size / 1e3:.0f} KB)")

resumo_referencias = pipeline.rodar_referencias(
    spark, lago, str(destino_ref / arquivos_ref[0]), str(destino_ref / arquivos_ref[1])
)
print(json.dumps(resumo_referencias, indent=2, ensure_ascii=False))
display(spark.table(lago.nome("gold_brent_semanal")).orderBy("semana", ascending=False).limit(8))

# COMMAND ----------

# MAGIC %md ## Coletas da ANP

# COMMAND ----------

pipeline.rodar_bronze(spark, lago, origem)
display(
    spark.table(lago.nome("bronze_precos")).groupBy("_arquivo").count().orderBy("_arquivo")
)

# COMMAND ----------

qualidade = pipeline.rodar_silver(lago)
print(json.dumps(qualidade, indent=2, ensure_ascii=False))

# COMMAND ----------

cobertura = pipeline.rodar_gold(lago)
print(json.dumps(cobertura, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Governança: tags e linhagem
# MAGIC A linhagem bronze → silver → gold aparece sozinha no Catalog Explorer (aba *Lineage*).
# MAGIC As tags abaixo marcam camada e fonte de cada tabela.

# COMMAND ----------

ANP, EIA, BCB = "ANP dados abertos", "US EIA via FRED", "Banco Central do Brasil"
camadas = {
    "bronze_precos": ("bronze", ANP),
    "silver_precos": ("silver", ANP),
    "silver_quarentena": ("silver", ANP),
    "bronze_brent": ("bronze", EIA),
    "bronze_dolar": ("bronze", BCB),
    "silver_referencias_diarias": ("silver", "US EIA e Banco Central"),
    "gold_brent_semanal": ("gold", "US EIA e Banco Central"),
    "gold_mercado_semanal": ("gold", ANP),
    "gold_bandeira_semanal": ("gold", ANP),
    "gold_nacional_semanal": ("gold", ANP),
    "gold_features": ("gold", "ANP US EIA e Banco Central"),
}
for tabela, (camada, fonte) in camadas.items():
    try:
        spark.sql(f"ALTER TABLE {lago.nome(tabela)} SET TAGS ('camada' = '{camada}', 'fonte' = '{fonte}')")
    except Exception as erro:  # tags podem não estar liberadas em todo workspace
        print(f"tag não aplicada em {tabela}: {erro}")

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {catalogo}.{esquema}"))
