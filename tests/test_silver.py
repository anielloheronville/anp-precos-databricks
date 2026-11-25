import datetime as dt

from anp_precos import silver

COLUNAS = [
    "regiao", "uf", "municipio", "revenda", "cnpj", "rua", "numero", "complemento", "bairro", "cep",
    "produto", "data_coleta", "valor_venda", "valor_compra", "unidade", "bandeira", "_arquivo",
]


def _linha(**kw):
    base = {
        "regiao": "CO", "uf": "MT", "municipio": "Sorriso", "revenda": "POSTO", "cnpj": "11.222.333/0001-81",
        "rua": "R", "numero": "1", "complemento": None, "bairro": "Centro", "cep": "78890-000",
        "produto": "GASOLINA", "data_coleta": "05/01/2026", "valor_venda": "6,49", "valor_compra": None,
        "unidade": "R$ / litro", "bandeira": "IPIRANGA", "_arquivo": "x.csv",
    }
    base.update(kw)
    return tuple(base[c] for c in COLUNAS)


def _df(spark, linhas):
    return spark.createDataFrame(linhas, schema=", ".join(f"{c} string" for c in COLUNAS))


def test_tipos_e_semana(spark):
    s, q = silver.separar(_df(spark, [_linha()]))
    r = s.collect()[0]
    assert q.count() == 0
    assert r["valor_venda"] == 6.49
    assert r["data_coleta"] == dt.date(2026, 1, 5)
    assert r["semana"] == dt.date(2026, 1, 5)  # segunda-feira
    assert r["cnpj"] == "11222333000181"
    assert r["municipio"] == "SORRISO"


def test_quarentena_com_motivo_sem_derrubar_a_leitura(spark):
    linhas = [
        _linha(produto="CENTRO"),  # coluna deslocada
        _linha(valor_venda="6,25,1"),  # número inválido: ANSI lançaria erro sem try_cast
        _linha(valor_venda="0,99"),
        _linha(data_coleta="31/02/2026"),
        _linha(cnpj="123"),
        _linha(uf="XX"),
        _linha(unidade="R$ / kg"),
    ]
    s, q = silver.separar(_df(spark, linhas))
    motivos = sorted(r["motivo_rejeicao"] for r in q.collect())
    assert s.count() == 0
    assert motivos == sorted([
        "produto_invalido", "preco_ausente", "preco_fora_da_faixa", "data_invalida",
        "cnpj_invalido", "uf_invalida", "unidade_invalida",
    ])


def test_gnv_em_metro_cubico_e_valido(spark):
    s, q = silver.separar(_df(spark, [_linha(produto="GNV", unidade="R$ / m³", valor_venda="4,79")]))
    assert q.count() == 0 and s.count() == 1


def test_gnv_com_m3_digitado_e_valido(spark):
    # o arquivo 2025-02 grava "R$ / m3" com o dígito 3
    s, q = silver.separar(_df(spark, [_linha(produto="GNV", unidade="R$ / m3", valor_venda="4,79")]))
    assert q.count() == 0
    assert s.collect()[0]["unidade"] == "R$ / m³"


def test_cnpj_alfanumerico_e_aceito(spark):
    s, q = silver.separar(_df(spark, [_linha(cnpj="12.ABC.345/01DE-35")]))
    assert q.count() == 0
    assert s.collect()[0]["cnpj"] == "12ABC34501DE35"


def test_linha_vazia_tem_motivo_proprio(spark):
    vazia = tuple(None if c != "_arquivo" else "x.csv" for c in COLUNAS)
    _, q = silver.separar(_df(spark, [vazia]))
    assert [r["motivo_rejeicao"] for r in q.collect()] == ["linha_vazia"]


def test_nomes_antigos_de_bandeira_sao_unificados(spark):
    linhas = [
        _linha(bandeira="VIBRA ENERGIA", cnpj="11222333000181"),
        _linha(bandeira="ALESAT", cnpj="11222333000182"),
        _linha(bandeira="RAIZEN MIME", cnpj="11222333000183"),
    ]
    s, _ = silver.separar(_df(spark, linhas))
    assert sorted(r["bandeira"] for r in s.collect()) == ["ALE", "RAIZEN", "VIBRA"]


def test_deduplica_mesma_coleta(spark):
    s, _ = silver.separar(_df(spark, [_linha(), _linha(_arquivo="y.csv")]))
    assert s.count() == 1


def test_bandeira_fora_das_principais_vira_outras(spark):
    s, _ = silver.separar(_df(spark, [_linha(bandeira="SABBÁ")]))
    r = s.collect()[0]
    assert r["bandeira"] == "OUTRAS" and r["bandeira_original"] == "SABBÁ"


def test_resumo_qualidade_fecha_a_conta(spark):
    bronze = _df(spark, [_linha(), _linha(), _linha(produto="X")])
    s, q = silver.separar(bronze)
    resumo = silver.resumo_qualidade(bronze, s, q)
    assert resumo == {
        "linhas_bronze": 3,
        "linhas_silver": 1,
        "linhas_quarentena": 1,
        "duplicatas_removidas": 1,
        "motivos": {"produto_invalido": 1},
    }
