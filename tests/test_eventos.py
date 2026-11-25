"""Estudo de eventos em pandas (sem Spark)."""

import numpy as np
import pandas as pd
import pytest

from anp_precos import eventos


def _mensal(n=96, semente=3, inicio="2017-01-01", ruido=0.02) -> pd.DataFrame:
    """Alimentos sobem 0,3% ao mês mais metade da variação do diesel do mês anterior."""
    rng = np.random.default_rng(semente)
    var_diesel = rng.normal(0, 3.0, n)
    var_mundo = rng.normal(0, 2.0, n)
    ipca_alimentos = 0.3 + 0.5 * np.concatenate([[0.0], var_diesel[:-1]]) + rng.normal(0, ruido, n)
    return pd.DataFrame(
        {
            "mes": pd.date_range(inicio, periods=n, freq="MS"),
            "dolar": 5.0,
            "barril_reais": 400.0,
            "gasolina": 6.0,
            "diesel_s10": 6 * np.exp(np.cumsum(var_diesel) / 100),
            "alimentos_mundo": 100 * np.exp(np.cumsum(var_mundo) / 100),
            "alimentos_mundo_reais": 500 * np.exp(np.cumsum(var_mundo) / 100),
            "ipca_alimentos": ipca_alimentos,
            "ipca_geral": 0.4,
        }
    )


def test_ipca_vira_indice_e_para_de_acumular_no_primeiro_buraco():
    tabela = _mensal(n=24)
    tabela.loc[10, "ipca_alimentos"] = np.nan
    dados = eventos.preparar(tabela)
    assert dados["nivel_ipca_geral"].iloc[0] == pytest.approx(100.4)
    assert dados["nivel_ipca_geral"].iloc[9] == pytest.approx(100 * 1.004**10)
    assert dados["nivel_ipca_geral"].notna().all()
    assert dados["nivel_ipca_alimentos"].iloc[:10].notna().all()
    assert dados["nivel_ipca_alimentos"].iloc[10:].isna().all()  # não emenda por cima do mês faltante


def test_ipca_que_comeca_depois_do_inicio_da_tabela_nao_anula_a_serie():
    # o download pode trazer o IPCA a partir de um mês posterior ao do dólar e do barril
    tabela = _mensal(n=36)
    tabela.loc[:4, ["ipca_geral", "ipca_alimentos"]] = np.nan
    dados = eventos.preparar(tabela)
    assert dados["nivel_ipca_geral"].iloc[:5].isna().all()
    assert dados["nivel_ipca_geral"].iloc[5] == pytest.approx(100.4)  # base 100 no mês anterior ao primeiro
    assert dados["nivel_ipca_geral"].iloc[5:].notna().all()
    resultado = eventos.tabela(tabela, eventos=[("Choque", "2018-06", "teste")], janelas=(3,)).set_index("serie")
    assert resultado.at["Inflação geral (IPCA)", "+3 meses"] == pytest.approx(100 * (1.004**3 - 1), abs=0.05)


def test_mes_ainda_sem_ipca_e_meses_antes_do_inicio_ficam_de_fora():
    tabela = _mensal(n=30, inicio="2016-07-01")
    tabela.loc[29, ["ipca_geral", "ipca_alimentos"]] = np.nan  # mês corrente: dólar parcial, IPCA não saiu
    dados = eventos.preparar(tabela)
    assert str(dados.index.min()) == "2017-01"  # o corte de início vale
    assert str(dados.index.max()) == str(tabela["mes"].iloc[28].to_period("M"))
    with pytest.raises(ValueError, match="nenhum mês com IPCA"):
        eventos.preparar(tabela.assign(ipca_geral=np.nan))


def test_tabela_de_eventos_mede_antes_e_depois_do_mes_base():
    tabela = _mensal(n=48)
    # rampa de 1% ao mês até o mês-base e um salto de 25% logo depois
    tabela["dolar"] = [4.0 * 1.01 ** (i - 23) for i in range(24)] + [5.0] * 24
    base = str(tabela["mes"].iloc[23].to_period("M"))
    resultado = eventos.tabela(tabela, eventos=[("Choque", base, "teste")], janelas=(3, 12)).set_index("serie")
    assert resultado.at["Dólar", "12 meses antes"] == pytest.approx(100 * (1.01**12 - 1), abs=0.05)  # 12, não 1
    assert resultado.at["Dólar", "+3 meses"] == 25.0
    assert resultado.at["Inflação geral (IPCA)", "+12 meses"] == pytest.approx(100 * (1.004**12 - 1), abs=0.05)
    assert resultado.at["Dólar", "evento"] == "Choque"


def test_janela_que_passa_do_fim_da_serie_fica_vazia():
    tabela = _mensal(n=30)
    base = str(tabela["mes"].iloc[27].to_period("M"))
    resultado = eventos.tabela(tabela, eventos=[("Recente", base, "teste")], janelas=(2, 12)).set_index("serie")
    assert resultado.at["Dólar", "+2 meses"] == 0.0
    assert pd.isna(resultado.at["Dólar", "+12 meses"])


def test_correlacao_aparece_na_defasagem_certa():
    correlacoes = eventos.correlacoes(_mensal(), defasagens=3)
    assert correlacoes.at["Diesel S10 na bomba", 1] > 0.95  # o diesel do mês anterior explica os alimentos
    assert abs(correlacoes.at["Diesel S10 na bomba", 0]) < 0.3
    assert abs(correlacoes.at["Alimentos no mundo (US$)", 1]) < 0.3


def test_canais_separam_intercepto_contribuicao_e_periodos():
    resultado = eventos.canais(_mensal(), defasagens=2)
    assert resultado["r2"] > 0.95
    # com ruído, o R² ajustado desconta os 6 regressores (2 canais x 3 defasagens)
    ruidoso = eventos.canais(_mensal(ruido=1.0), defasagens=2)
    n = ruidoso["meses"]
    assert 0.3 < ruidoso["r2_ajustado"] < ruidoso["r2"] < 0.95
    assert ruidoso["r2_ajustado"] == pytest.approx(1 - (1 - ruidoso["r2"]) * (n - 1) / (n - 6 - 1), abs=2e-3)
    assert resultado["intercepto_pct_ao_mes"] == pytest.approx(0.3, abs=0.02)
    assert resultado["soma_dos_coeficientes"]["Diesel S10 na bomba"] == pytest.approx(0.5, abs=0.05)
    assert abs(resultado["soma_dos_coeficientes"]["Alimentos no mundo (R$)"]) < 0.05
    assert resultado["meses_com_canal_interpolado"] == 0
    periodos = resultado["por_periodo"]
    assert list(periodos) == list(eventos.PERIODOS)
    # fronteiras: 2017-01 a 2020-02 são 38 meses; mar/2020 a dez/2022, 34; de 2023 a dez/2024, 24
    assert [p["meses"] for p in periodos.values()] == [38, 34, 24]
    assert periodos["antes da COVID (até fev/2020)"]["meses_na_regressao"] == 38 - 3  # variação + 2 defasagens
    for periodo in periodos.values():
        # alimentos = intercepto + contribuição dos canais (o ruído sintético é desprezível)
        esperado = periodo["alimentos_pct_ao_mes"] - 0.3
        assert periodo["contribuicao_dos_canais_pct_ao_mes"] == pytest.approx(esperado, abs=0.12)


def test_um_mes_sem_diesel_e_interpolado_e_nao_tira_meses_da_conta():
    # a ANP não pesquisou preços em set/2020: sem interpolar, cairiam esse mês e os das defasagens
    tabela = _mensal()
    posicao = 44  # set/2020
    assert str(tabela["mes"].iloc[posicao].date()) == "2020-09-01"
    furada = tabela.copy()
    furada.loc[posicao, "diesel_s10"] = np.nan
    inteiro, com_buraco = eventos.canais(tabela, defasagens=4), eventos.canais(furada, defasagens=4)
    assert com_buraco["meses_com_canal_interpolado"] == 1
    assert com_buraco["meses"] == inteiro["meses"]
    periodo = "COVID e Ucrânia (mar/2020 a dez/2022)"
    assert com_buraco["por_periodo"][periodo]["meses_na_regressao"] == 34
    # a inflação do período é a de todos os meses, com ou sem buraco no canal
    antes, depois = inteiro["por_periodo"][periodo], com_buraco["por_periodo"][periodo]
    assert depois["alimentos_pct_ao_mes"] == antes["alimentos_pct_ao_mes"]
    # buraco de dois meses seguidos não é interpolado
    furada.loc[posicao + 1, "diesel_s10"] = np.nan
    assert eventos.canais(furada, defasagens=4)["meses"] < inteiro["meses"]
