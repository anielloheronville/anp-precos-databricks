# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze → Silver → Gold
# MAGIC
# MAGIC O mesmo código de `src/anp_precos` que roda (e é testado) localmente, aqui gravando
# MAGIC tabelas Delta no Unity Catalog, com comentário em cada tabela e liquid clustering
# MAGIC por produto e semana nas tabelas grandes.

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

from anp_precos import pipeline  # noqa: E402
from anp_precos.lago import Lago  # noqa: E402

dbutils.widgets.text("catalogo", "workspace")
dbutils.widgets.text("esquema", "anp")
catalogo = dbutils.widgets.get("catalogo")
esquema = dbutils.widgets.get("esquema")
origem = f"/Volumes/{catalogo}/{esquema}/bruto/csv/"
lago = Lago(spark, catalogo=catalogo, esquema=esquema)

# COMMAND ----------

pipeline.rodar_bronze(spark, lago, origem)
display(
    spark.table(lago.nome("bronze_precos")).groupBy("_arquivo").count().orderBy("_arquivo")
)

# COMMAND ----------

qualidade = pipeline.rodar_silver(lago)
print(json.dumps(qualidade, indent=2, ensure_ascii=False))

# COMMAND ----------

pipeline.rodar_gold(lago)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Governança: tags e linhagem
# MAGIC A linhagem bronze → silver → gold aparece sozinha no Catalog Explorer (aba *Lineage*).
# MAGIC As tags abaixo marcam camada e fonte de cada tabela.

# COMMAND ----------

camadas = {
    "bronze_precos": "bronze",
    "silver_precos": "silver",
    "silver_quarentena": "silver",
    "gold_mercado_semanal": "gold",
    "gold_bandeira_semanal": "gold",
    "gold_nacional_semanal": "gold",
    "gold_features": "gold",
}
for tabela, camada in camadas.items():
    try:
        spark.sql(f"ALTER TABLE {lago.nome(tabela)} SET TAGS ('camada' = '{camada}', 'fonte' = 'ANP dados abertos')")
    except Exception as erro:  # tags podem não estar liberadas em todo workspace
        print(f"tag não aplicada em {tabela}: {erro}")

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {catalogo}.{esquema}"))
