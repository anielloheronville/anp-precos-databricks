import datetime as dt

import pytest

from anp_precos import gold

ESQUEMA = (
    "regiao string, uf string, municipio string, cnpj string, produto string, "
    "semana date, valor_venda double, bandeira string"
)


def _coleta(cnpj, preco, bandeira, semana=dt.date(2026, 1, 5), municipio="SORRISO"):
    return ("CO", "MT", municipio, cnpj, "GASOLINA", semana, preco, bandeira)


def test_premio_ipiranga_contra_concorrentes_do_mesmo_municipio(spark):
    silver = spark.createDataFrame(
        [
            _coleta("1", 6.60, "IPIRANGA"),
            _coleta("2", 6.00, "VIBRA"),
            _coleta("3", 6.00, "BRANCA"),
            _coleta("4", 6.00, "RAIZEN"),
        ],
        ESQUEMA,
    )
    r = gold.mercado_semanal(silver).collect()[0]
    assert r["n_postos"] == 4 and r["n_ipiranga"] == 1 and r["n_outros"] == 3
    assert r["preco_mediano_ipiranga"] == 6.60
    assert r["preco_mediano_outros"] == 6.00
    assert r["premio_ipiranga_pct"] == pytest.approx(10.0)


def test_mediana_exata_com_numero_par_de_postos(spark):
    # percentile_approx devolveria 6,00 vs 6,10 (prêmio -1,6%); a mediana exata dá 6,10 vs 6,10
    silver = spark.createDataFrame(
        [
            _coleta("1", 6.00, "IPIRANGA"),
            _coleta("2", 6.20, "IPIRANGA"),
            _coleta("3", 6.00, "VIBRA"),
            _coleta("4", 6.10, "BRANCA"),
            _coleta("5", 6.10, "RAIZEN"),
            _coleta("6", 6.20, "ALE"),
        ],
        ESQUEMA,
    )
    r = gold.mercado_semanal(silver).collect()[0]
    assert r["preco_mediano_ipiranga"] == pytest.approx(6.10)
    assert r["preco_mediano_outros"] == pytest.approx(6.10)
    assert r["premio_ipiranga_pct"] == pytest.approx(0.0, abs=1e-9)
    assert r["preco_mediano"] == pytest.approx(6.10)


def test_sem_concorrentes_suficientes_nao_calcula_premio(spark):
    silver = spark.createDataFrame([_coleta("1", 6.6, "IPIRANGA"), _coleta("2", 6.0, "VIBRA")], ESQUEMA)
    r = gold.mercado_semanal(silver).collect()[0]
    assert r["premio_ipiranga_pct"] is None


def _mercado(spark, semanas_precos):
    linhas = [_coleta(str(i), p, "VIBRA", semana=s) for i, (s, p) in enumerate(semanas_precos)]
    return gold.mercado_semanal(spark.createDataFrame(linhas, ESQUEMA))


def test_features_respeitam_semana_faltante(spark):
    s0 = dt.date(2026, 1, 5)
    semanas = [
        (s0, 6.00),
        (s0 + dt.timedelta(days=7), 6.10),
        # semana 3 não foi visitada pela ANP
        (s0 + dt.timedelta(days=21), 6.30),
    ]
    mercado = _mercado(spark, semanas)
    nacional = gold.nacional_semanal(
        spark.createDataFrame([_coleta(str(i), p, "VIBRA", semana=s) for i, (s, p) in enumerate(semanas)], ESQUEMA)
    )
    linhas = {r["semana"]: r for r in gold.features(mercado, nacional).collect()}

    primeira, segunda, quarta = (linhas[s] for s, _ in semanas)
    assert primeira["var_1s"] is None
    assert primeira["sobe_prox_semana"] == 1  # 6,00 -> 6,10 na semana seguinte
    assert segunda["var_1s"] == pytest.approx(6.10 / 6.00 - 1)
    assert segunda["sobe_prox_semana"] is None  # a semana seguinte não existe
    assert quarta["var_1s"] is None  # não compara com 2 semanas atrás fingindo ser 1


BRENT = "semana date, var_brent_1s double, var_brent_4s double, var_brent_8s double"


def test_brent_entra_com_uma_semana_de_atraso(spark):
    """A fonte publica o Brent com ~1 semana de atraso: a linha de t só enxerga o barril até t-1."""
    s = [dt.date(2026, 1, 5) + dt.timedelta(days=7 * i) for i in range(4)]
    semanas = [(s[1], 6.00), (s[2], 6.10), (s[3], 6.20)]
    mercado = _mercado(spark, semanas)
    nacional = gold.nacional_semanal(
        spark.createDataFrame([_coleta(str(i), p, "VIBRA", semana=d) for i, (d, p) in enumerate(semanas)], ESQUEMA)
    )
    # a tabela do Brent tem s0, s1 e s2; falta s2 -> a linha de s3 fica sem barril, mas continua na tabela
    brent = spark.createDataFrame([(s[0], 0.01, 0.04, 0.08), (s[1], 0.30, 0.34, 0.38)], BRENT)
    linhas = {r["semana"]: r for r in gold.features(mercado, nacional, brent=brent).collect()}

    assert set(linhas) == {s[1], s[2], s[3]}  # a junção não perde nem duplica linhas
    assert linhas[s[1]]["var_brent_1s"] == pytest.approx(0.01)  # barril de s0, nunca o de s1
    assert linhas[s[1]]["var_brent_8s"] == pytest.approx(0.08)
    assert linhas[s[2]]["var_brent_4s"] == pytest.approx(0.34)  # barril de s1
    assert linhas[s[3]]["var_brent_1s"] is None


def test_sem_tabela_do_brent_as_colunas_existem_e_ficam_nulas(spark):
    s0 = dt.date(2026, 1, 5)
    mercado = _mercado(spark, [(s0, 6.00)])
    nacional = gold.nacional_semanal(spark.createDataFrame([_coleta("1", 6.0, "VIBRA", semana=s0)], ESQUEMA))
    sem = gold.features(mercado, nacional)
    com = gold.features(mercado, nacional, brent=spark.createDataFrame([], BRENT))
    linha = sem.collect()[0]
    assert linha["var_brent_1s"] is None and linha["var_brent_4s"] is None and linha["var_brent_8s"] is None
    assert sem.columns == com.columns  # mesmo esquema nos dois caminhos


def test_alvo_usa_limiar(spark):
    s0 = dt.date(2026, 1, 5)
    mercado = _mercado(spark, [(s0, 6.00), (s0 + dt.timedelta(days=7), 6.02)])  # +0,33%
    nacional = gold.nacional_semanal(
        spark.createDataFrame([_coleta("1", 6.0, "VIBRA", semana=s0)], ESQUEMA)
    )
    linhas = sorted(gold.features(mercado, nacional).collect(), key=lambda r: r["semana"])
    assert linhas[0]["sobe_prox_semana"] == 0
