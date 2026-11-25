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
