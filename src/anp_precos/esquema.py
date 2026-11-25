"""Contrato dos dados: nomes de colunas, domínios válidos e faixas de preço."""

# Cabeçalho original da ANP -> nome de coluna sem espaço (Delta não aceita espaço sem column mapping).
COLUNAS_BRUTAS = {
    "Regiao - Sigla": "regiao",
    "Estado - Sigla": "uf",
    "Municipio": "municipio",
    "Revenda": "revenda",
    "CNPJ da Revenda": "cnpj",
    "Nome da Rua": "rua",
    "Numero Rua": "numero",
    "Complemento": "complemento",
    "Bairro": "bairro",
    "Cep": "cep",
    "Produto": "produto",
    "Data da Coleta": "data_coleta",
    "Valor de Venda": "valor_venda",
    "Valor de Compra": "valor_compra",
    "Unidade de Medida": "unidade",
    "Bandeira": "bandeira",
}

PRODUTOS = ["GASOLINA", "GASOLINA ADITIVADA", "ETANOL", "DIESEL", "DIESEL S10", "GNV"]
UNIDADES = ["R$ / litro", "R$ / m³"]
REGIOES = ["N", "NE", "CO", "SE", "S"]
UFS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]

# Bandeiras com volume para análise própria; o resto vira OUTRAS.
BANDEIRAS_PRINCIPAIS = ["IPIRANGA", "VIBRA", "RAIZEN", "BRANCA", "ALE"]

# A ANP trocou nomes no meio da série (VIBRA ENERGIA -> VIBRA em 2023, ALESAT -> ALE em 2024).
# RAIZEN MIME é bandeira do grupo Raízen; aqui conta como RAIZEN (decisão documentada no README).
APELIDOS_BANDEIRA = {
    "VIBRA ENERGIA": "VIBRA",
    "ALESAT": "ALE",
    "RAIZEN MIME": "RAIZEN",
}
BANDEIRA_FOCO = "IPIRANGA"

# Faixa plausível de preço de bomba (R$/litro ou R$/m³). Fora disso é erro de digitação ou de coluna.
PRECO_MIN = 2.0
PRECO_MAX = 12.0

# Variação semanal da mediana municipal que conta como "alta" no alvo do modelo.
LIMIAR_ALTA = 0.005

# Séries de referência (Brent e dólar): formato de cada fonte, como o arquivo chega.
FONTES_REFERENCIA = {
    # U.S. EIA, série DCOILBRENTEU do FRED: "2026-03-09,118.40"; feriado vem com o valor vazio
    # faixa: valor fora dela é erro de arquivo (o Brent spot nunca saiu de 9 a 144 US$ desde 1987)
    "brent": {
        "sep": ",",
        "colunas": {"observation_date": "data", "DCOILBRENTEU": "valor"},
        "formato_data": "yyyy-MM-dd",
        "faixa": (5.0, 300.0),
    },
    # Banco Central, série 1 do SGS (PTAX venda): '"09/03/2026";"5,1234"'
    "dolar": {
        "sep": ";",
        "colunas": {"data": "data", "valor": "valor"},
        "formato_data": "dd/MM/yyyy",
        "faixa": (0.5, 20.0),
    },
}
# Séries mensais usadas no estudo de eventos (alimentos): mesmo formato das duas fontes acima.
FONTES_REFERENCIA.update(
    {
        # IPCA, variação mensal em % (IBGE, pelas séries 433 e 1635 do SGS do Banco Central)
        "ipca_geral": {**FONTES_REFERENCIA["dolar"], "faixa": (-5.0, 10.0)},
        "ipca_alimentos": {**FONTES_REFERENCIA["dolar"], "faixa": (-5.0, 10.0)},
        # Índice mundial de preços de alimentos em dólares (FMI, série PFOODINDEXM do FRED)
        "alimentos_mundo": {
            "sep": ",",
            "colunas": {"observation_date": "data", "PFOODINDEXM": "valor"},
            "formato_data": "yyyy-MM-dd",
            "faixa": (20.0, 500.0),
        },
    }
)
ARQUIVOS_REFERENCIA = {
    "brent": "brent_usd_diario.csv",
    "dolar": "dolar_ptax_venda_diario.csv",
    "ipca_geral": "ipca_geral_mensal.csv",
    "ipca_alimentos": "ipca_alimentos_mensal.csv",
    "alimentos_mundo": "alimentos_mundo_mensal.csv",
    "bomba_brasil": "bomba_brasil_semanal.csv",
}

# Preço médio de revenda no Brasil por semana (ANP, série histórica do levantamento de preços).
# O CSV versionado é a planilha da ANP convertida por scripts/baixar_referencias.py.
COLUNAS_BOMBA = ["data_inicial", "data_final", "produto", "numero_de_postos", "unidade", "preco_medio_revenda"]
CABECALHO_PLANILHA_BOMBA = {
    "DATA INICIAL": "data_inicial",
    "DATA FINAL": "data_final",
    "PRODUTO": "produto",
    "NÚMERO DE POSTOS PESQUISADOS": "numero_de_postos",
    "UNIDADE DE MEDIDA": "unidade",
    "PREÇO MÉDIO REVENDA": "preco_medio_revenda",
}
# nome do produto na planilha semanal -> nome usado no resto do projeto
PRODUTOS_BOMBA = {
    "GASOLINA COMUM": "GASOLINA",
    "GASOLINA ADITIVADA": "GASOLINA ADITIVADA",
    "ETANOL HIDRATADO": "ETANOL",
    "OLEO DIESEL": "DIESEL",
    "OLEO DIESEL S10": "DIESEL S10",
    "GNV": "GNV",
    "GLP": "GLP",
}
FAIXA_PRECO_BOMBA = (0.5, 300.0)  # de R$/litro a R$/botijão de 13 kg
# produtos acompanhados no repasse do barril, e o nome da coluna de cada um na tabela semanal
PRODUTOS_REPASSE = {"GASOLINA": "gasolina", "DIESEL S10": "diesel_s10", "DIESEL": "diesel", "ETANOL": "etanol"}

# Em 16/05/2023 a Petrobras anunciou o fim da paridade de importação; a segunda-feira daquela semana
# separa os dois regimes de repasse.
CORTE_PARIDADE = "2023-05-15"
DEFASAGENS_REPASSE = 16  # semanas

# Eventos do estudo: (nome, mês-base = último mês antes do choque, o que aconteceu). Datas públicas.
EVENTOS = [
    ("COVID", "2020-02", "pandemia declarada em 11/03/2020"),
    ("Ucrânia", "2022-01", "invasão em 24/02/2022"),
    ("Gaza", "2023-09", "ataque de 07/10/2023 e guerra em Gaza"),
    ("Irã (12 dias)", "2025-05", "ataques de 13/06/2025"),
    ("Irã (Ormuz)", "2026-02", "ataques de 28/02/2026 e tráfego interrompido no Estreito de Ormuz"),
]

LITROS_POR_BARRIL = 158.987
# Semana com menos cotações que isso (feriados, arquivo cortado) fica sem preço, em vez de média de 1 dia.
MIN_DIAS_SEMANA = 3
# Horizontes, em semanas, das variações do barril em reais usadas como feature.
PASSOS_BRENT = (1, 4, 8)
COLUNAS_BRENT = [f"var_brent_{passos}s" for passos in PASSOS_BRENT]
# A EIA publica o Brent spot com cerca de uma semana de atraso: ao fim da semana t, a última semana
# completa na fonte é t-1. A linha da semana t usa o barril até t-1, que é o que existiria na hora.
DEFASAGEM_BRENT_SEMANAS = 1
# Se o barril faltar em mais do que esta fração das semanas do gold_features, a carga falha.
MAX_SEMANAS_SEM_BRENT = 0.10
