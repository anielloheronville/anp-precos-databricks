# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · O barril chega à bomba? Repasse, cenários e eventos
# MAGIC
# MAGIC > Estudo independente sobre dados públicos, sem relação com nenhuma distribuidora.
# MAGIC
# MAGIC 1. **Repasse:** quanto de uma alta do barril de Brent em reais chega ao preço médio na bomba, e em
# MAGIC    quantas semanas, antes e depois de a Petrobras deixar a paridade de importação (mai/2023).
# MAGIC 2. **Cenários:** o que essa medida implica para a gasolina e o diesel se o Brent e o dólar forem
# MAGIC    para níveis que a própria série já mostrou. São contas condicionais, não previsões.
# MAGIC 3. **Eventos:** o que aconteceu com dólar, barril, diesel e alimentos depois da COVID e das guerras
# MAGIC    da Ucrânia, de Gaza e do Irã.
# MAGIC
# MAGIC As séries (Brent, dólar, preço semanal da ANP, IPCA e alimentos no mundo) vêm de `referencias/`
# MAGIC no repositório. Este notebook copia os arquivos para o volume e recria as tabelas dessas séries,
# MAGIC então só depende do notebook 00 (que cria o schema e o volume), não do 01 nem do 02.

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn>=1.7" scipy matplotlib

# COMMAND ----------

import json
import os
import shutil
import sys
from pathlib import Path

RAIZ = Path(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, str(RAIZ / "src"))

from IPython.display import Image  # noqa: E402
from IPython.display import display as mostrar  # noqa: E402

from anp_precos import eventos, graficos, pipeline, repasse  # noqa: E402
from anp_precos.esquema import ARQUIVOS_REFERENCIA, CORTE_PARIDADE, EVENTOS  # noqa: E402
from anp_precos.lago import Lago  # noqa: E402

dbutils.widgets.text("catalogo", "workspace")
dbutils.widgets.text("esquema", "anp")
catalogo = dbutils.widgets.get("catalogo")
esquema = dbutils.widgets.get("esquema")
volume = f"/Volumes/{catalogo}/{esquema}/bruto"
lago = Lago(spark, catalogo=catalogo, esquema=esquema)

figuras = Path("/tmp/anp_figuras_repasse")
shutil.rmtree(figuras, ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Séries de referência
# MAGIC Bronze como veio de cada fonte, silver tipado, gold por semana e por mês. Linha que não é data e
# MAGIC número na faixa esperada derruba a carga.

# COMMAND ----------

destino_ref = Path(volume) / "referencias"
destino_ref.mkdir(exist_ok=True)
caminhos = {}
for serie, nome in ARQUIVOS_REFERENCIA.items():
    fonte = RAIZ / "referencias" / nome
    if not fonte.exists():
        raise FileNotFoundError(f"{fonte} não existe: faça pull do Git folder na branch que tem a pasta referencias/.")
    shutil.copyfile(fonte, destino_ref / nome)
    caminhos[serie] = str(destino_ref / nome)

resumo = {
    "brent_e_dolar": pipeline.rodar_referencias(spark, lago, caminhos["brent"], caminhos["dolar"]),
    "bomba_e_series_mensais": pipeline.rodar_macro(spark, lago, caminhos),
}
print(json.dumps(resumo, indent=2, ensure_ascii=False))

# COMMAND ----------

semanal = lago.ler("gold_repasse_semanal").toPandas()
mensal = lago.ler("gold_macro_mensal").toPandas()
print(f"{len(semanal):,} semanas · {len(mensal):,} meses")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Barril e bomba
# MAGIC Tudo em reais por litro, na mesma escala: o barril de Brent convertido pelo dólar e dividido por
# MAGIC 159 litros, ao lado do preço médio de revenda no Brasil.

# COMMAND ----------

mostrar(Image(str(graficos.barril_e_bomba(semanal, figuras, corte=CORTE_PARIDADE))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quanto do barril chega à bomba
# MAGIC Regressão das variações semanais da bomba nas variações do barril da mesma semana e das 16
# MAGIC anteriores. A curva é o repasse acumulado de 1% de alta do barril; a faixa é o intervalo de 90%
# MAGIC por bootstrap em blocos de 8 semanas. A conta não separa mudanças de imposto.

# COMMAND ----------

analise = repasse.relatorio(semanal)
mostrar(Image(str(graficos.curvas_de_repasse(analise["curvas"], figuras))))
print(json.dumps(repasse.resumo_das_curvas(analise["curvas"]), indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cenários
# MAGIC As hipóteses saem da própria série: o Brent volta ao pico ou ao piso dos últimos 12 meses, e o
# MAGIC dólar de estresse é a maior média semanal da série. O barril salta para o nível da hipótese na
# MAGIC semana seguinte e fica lá; entra também o repasse ainda pendente das variações que o barril já teve.
# MAGIC
# MAGIC Para a gasolina há duas leituras: com o repasse medido de mai/2023 para cá, e com o repasse do
# MAGIC período de paridade de importação. A diferença entre as duas é a política de preços.

# COMMAND ----------

print(json.dumps(analise["situacao_atual"], indent=2, ensure_ascii=False))
display(analise["cenarios"][analise["cenarios"]["semanas_depois"] == 16])
mostrar(Image(str(graficos.cenarios(analise["cenarios"], figuras))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## COVID, guerras e alimentos
# MAGIC Variação de cada série nos 12 meses antes do mês-base de cada evento e nos 3, 6 e 12 meses
# MAGIC seguintes. É descritivo: mostra o que mudou depois de cada data, não prova a causa.

# COMMAND ----------

display(eventos.tabela(mensal).drop(columns=["descricao"]))
mostrar(Image(str(graficos.eventos_e_alimentos(eventos.preparar(mensal), EVENTOS, figuras))))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Por quais canais os alimentos andam
# MAGIC Correlação do IPCA de alimentos do mês com a variação de cada série k meses antes, e uma regressão
# MAGIC nos dois canais com sinal (alimentos no mundo em reais e diesel). Por período, a regressão separa
# MAGIC a contribuição dos canais do intercepto, que é a inflação de alimentos que não depende deles.
# MAGIC
# MAGIC O bloco `aceleracao` diz quanto da alta dos alimentos entre o período anterior à COVID e
# MAGIC 2020 a 2022 os canais acompanham: duas estimativas (com e sem o nível de cada período
# MAGIC controlado), cada uma com dois intervalos de 90% (a conta inteira refeita em reamostras dos
# MAGIC meses, e só os coeficientes), e a mesma conta em cada combinação de número de defasagens e
# MAGIC de mês em que o período do choque termina. O resultado é uma faixa larga, não um ponto, e
# MAGIC não prova causa.

# COMMAND ----------

display(eventos.correlacoes(mensal).rename(columns=str).reset_index())
print(json.dumps(eventos.canais(mensal), indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md ## Governança: tags das tabelas deste notebook

# COMMAND ----------

ANP, IBGE, FMI = "ANP dados abertos", "IBGE pelo SGS do Banco Central", "FMI via FRED"
camadas = {
    "bronze_bomba_brasil": ("bronze", ANP),
    "bronze_ipca_geral": ("bronze", IBGE),
    "bronze_ipca_alimentos": ("bronze", IBGE),
    "bronze_alimentos_mundo": ("bronze", FMI),
    "silver_bomba_brasil": ("silver", ANP),
    "gold_repasse_semanal": ("gold", "ANP US EIA e Banco Central"),
    "gold_macro_mensal": ("gold", "ANP IBGE FMI US EIA e Banco Central"),
}
for tabela, (camada, fonte) in camadas.items():
    try:
        spark.sql(f"ALTER TABLE {lago.nome(tabela)} SET TAGS ('camada' = '{camada}', 'fonte' = '{fonte}')")
    except Exception as erro:  # tags podem não estar liberadas em todo workspace
        print(f"tag não aplicada em {tabela}: {erro}")

display(spark.sql(f"SHOW TABLES IN {catalogo}.{esquema}"))
