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
