# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Prêmio da Ipiranga e modelo de alta de preço
# MAGIC
# MAGIC > Estudo independente sobre dados públicos da ANP, sem relação com nenhuma distribuidora.
# MAGIC > Bandeira conforme declarada à ANP; as marcas pertencem aos seus titulares.
# MAGIC
# MAGIC 1. **Prêmio:** em cada município e semana, quanto a mediana dos postos Ipiranga fica acima
# MAGIC    ou abaixo da mediana dos concorrentes ao lado (bootstrap por município, SciPy).
# MAGIC 2. **Modelo:** a mediana do município sobe ≥ 0,5% na semana que vem? Gradient boosting
# MAGIC    (scikit-learn) com validação no tempo, comparado com persistência e regressão logística.
# MAGIC    Parâmetros, métricas, gráficos e o modelo ficam registrados no MLflow.

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn>=1.7" scipy matplotlib

# COMMAND ----------

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

import mlflow  # noqa: E402

from anp_precos import estatistica, graficos, modelo  # noqa: E402
from anp_precos.esquema import LIMIAR_ALTA  # noqa: E402
from anp_precos.lago import Lago  # noqa: E402

dbutils.widgets.text("catalogo", "workspace")
dbutils.widgets.text("esquema", "anp")
catalogo = dbutils.widgets.get("catalogo")
esquema = dbutils.widgets.get("esquema")
lago = Lago(spark, catalogo=catalogo, esquema=esquema)

figuras = Path("/tmp/anp_figuras")
shutil.rmtree(figuras, ignore_errors=True)  # só as figuras desta execução vão para o MLflow
mlflow.sklearn.autolog(disable=True)  # o run abaixo registra só o que importa

# COMMAND ----------

mercado = lago.ler("gold_mercado_semanal").toPandas()
por_bandeira = lago.ler("gold_bandeira_semanal").toPandas()
feats = lago.ler("gold_features").toPandas()
print(f"{len(mercado):,} linhas de mercado · {len(feats):,} linhas de features")

# COMMAND ----------

premio = estatistica.premio_por_produto(mercado)
display(premio)

# COMMAND ----------

for produto in ["GASOLINA", "DIESEL S10", "ETANOL"]:
    graficos.evolucao_por_bandeira(por_bandeira, produto, figuras)
    graficos.premio_por_uf(mercado, produto, figuras)

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/anp-alta-de-preco")
with mlflow.start_run(run_name="hist_gradient_boosting"):
    resultado = modelo.treinar_e_avaliar(feats)
    importancias = modelo.importancia_por_permutacao(resultado)
    mlflow.log_params({"limiar_alta": LIMIAR_ALTA, **resultado["cortes"]})
    for particao in ["validacao", "teste"]:
        for nome, metricas in resultado[particao].items():
            for metrica, valor in metricas.items():
                mlflow.log_metric(f"{particao}_{nome}_{metrica}", valor)
    mlflow.log_dict(importancias, "importancia_permutacao.json")
    mlflow.log_dict(premio.to_dict(orient="records"), "premio_ipiranga.json")
    graficos.curva_roc(resultado, figuras)
    graficos.importancia(importancias, figuras)
    mlflow.log_artifacts(str(figuras), artifact_path="figuras")
    exemplo = resultado["_x_teste"].head(5)
    try:  # MLflow 3
        mlflow.sklearn.log_model(resultado["_modelo"], name="modelo", input_example=exemplo)
    except TypeError:  # MLflow 2
        mlflow.sklearn.log_model(resultado["_modelo"], artifact_path="modelo", input_example=exemplo)

{k: v for k, v in resultado.items() if not k.startswith("_")}

# COMMAND ----------

display(spark.createDataFrame([{"feature": k, "queda_auc": v} for k, v in importancias.items()]))
