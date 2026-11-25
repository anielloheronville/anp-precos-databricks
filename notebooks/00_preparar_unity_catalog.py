# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Preparar o Unity Catalog
# MAGIC
# MAGIC 1. Cria o schema `anp` e o volume `bruto` (na Free Edition o catálogo padrão é `workspace`).
# MAGIC 2. **Você sobe os zips da ANP:** Catalog → workspace → anp → Volumes → bruto →
# MAGIC    *Upload to this volume* → os arquivos `ca-AAAA-SS.zip` baixados de
# MAGIC    [dados abertos da ANP](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis).
# MAGIC 3. Rode a última célula: ela descompacta cada zip para `anp_precos_AAAA_SS.csv` no próprio volume.

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace")
dbutils.widgets.text("esquema", "anp")
catalogo = dbutils.widgets.get("catalogo")
esquema = dbutils.widgets.get("esquema")
volume = f"/Volumes/{catalogo}/{esquema}/bruto"

# COMMAND ----------

spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {catalogo}.{esquema} "
    "COMMENT 'Radar de preços de combustíveis (dados abertos da ANP)'"
)
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalogo}.{esquema}.bruto COMMENT 'Arquivos da ANP como baixados'")
display(spark.sql(f"SHOW VOLUMES IN {catalogo}.{esquema}"))

# COMMAND ----------

# MAGIC %md ### Descompactar (rode depois do upload)

# COMMAND ----------

import re
import shutil
import zipfile
from pathlib import Path

pasta = Path(volume)
csv_dir = pasta / "csv"
csv_dir.mkdir(exist_ok=True)
for arquivo in sorted(pasta.glob("*.zip")):
    with zipfile.ZipFile(arquivo) as z:
        for membro in z.infolist():
            achado = re.search(r"(\d{4})\.(\d{2})\.csv$", membro.filename)
            if not achado:
                continue
            destino = csv_dir / f"anp_precos_{achado.group(1)}_{achado.group(2)}.csv"
            with z.open(membro) as origem, open(destino, "wb") as saida:
                shutil.copyfileobj(origem, saida, length=16 * 1024 * 1024)
            print(f"{arquivo.name} -> {destino.name} ({destino.stat().st_size / 1e6:.0f} MB)")

display(dbutils.fs.ls(f"{volume}/csv/"))
