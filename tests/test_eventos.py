"""Estudo de eventos em pandas (sem Spark)."""

import inspect

import numpy as np
import pandas as pd
import pytest

from anp_precos import eventos

SIMPLES, CONTROLADA = "sem_controle_de_periodo", "com_controle_de_periodo"
CORTES = ("2022-06", "2022-12", "2023-06")
DIESEL, MUNDO = "Diesel S10 na bomba", "Alimentos no mundo (R$)"


def _mensal(
    n=96,
    semente=3,
    inicio="2017-01-01",
    ruido=0.02,
    diesel_no_choque=0.0,
    extra_no_choque=0.0,
    atraso=1,
    fim_do_choque="2022-12-01",
) -> pd.DataFrame:
    """Alimentos sobem 0,3% ao mês mais metade da variação do diesel de `atraso` meses antes.

    De mar/2020 a `fim_do_choque`, o diesel pode subir `diesel_no_choque` a mais por mês, e os
    alimentos podem ganhar `extra_no_choque` que nenhum canal explica.
    """
    rng = np.random.default_rng(semente)
    meses = pd.date_range(inicio, periods=n, freq="MS")
    choque = (meses >= "2020-03-01") & (meses <= fim_do_choque)
    var_diesel = rng.normal(0, 3.0, n) + diesel_no_choque * choque
    var_mundo = rng.normal(0, 2.0, n)
    ipca_alimentos = 0.3 + 0.5 * np.concatenate([np.zeros(atraso), var_diesel[:-atraso]]) + rng.normal(0, ruido, n)
    ipca_alimentos = ipca_alimentos + extra_no_choque * choque
    return pd.DataFrame(
        {
            "mes": meses,
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
    resultado = eventos.canais(_mensal(), defasagens=2, n_boot=20)
    assert resultado["r2"] > 0.95
    # com ruído, o R² ajustado desconta os 6 regressores (2 canais x 3 defasagens)
    ruidoso = eventos.canais(_mensal(ruido=1.0), defasagens=2, n_boot=20)
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
    inteiro = eventos.canais(tabela, defasagens=4, n_boot=20)
    com_buraco = eventos.canais(furada, defasagens=4, n_boot=20)
    assert com_buraco["meses_com_canal_interpolado"] == 1
    assert com_buraco["meses"] == inteiro["meses"]
    periodo = "COVID e Ucrânia (mar/2020 a dez/2022)"
    assert com_buraco["por_periodo"][periodo]["meses_na_regressao"] == 34
    # a inflação do período é a de todos os meses, com ou sem buraco no canal
    antes, depois = inteiro["por_periodo"][periodo], com_buraco["por_periodo"][periodo]
    assert depois["alimentos_pct_ao_mes"] == antes["alimentos_pct_ao_mes"]
    # buraco de dois meses seguidos não é interpolado
    furada.loc[posicao + 1, "diesel_s10"] = np.nan
    assert eventos.canais(furada, defasagens=4, n_boot=20)["meses"] < inteiro["meses"]


# ---- quanto da aceleração dos alimentos os canais acompanham


def _a_mao(tabela: pd.DataFrame, defasagens: int = 2, fim: str = "2022-12", atraso: int = 1) -> dict:
    """A conta refeita sem o módulo, com o coeficiente verdadeiro do dado sintético.

    A contribuição do diesel é 0,5 x a variação de `atraso` meses antes. A fração é a diferença das
    médias dessa contribuição entre os dois períodos, dividida pela aceleração dos alimentos, tudo
    nos meses que entram na regressão (a variação custa um mês e cada defasagem, mais um).
    """
    mes = tabela["mes"].dt.to_period("M")
    contribuicao = 0.5 * (100 * np.log(tabela["diesel_s10"] / tabela["diesel_s10"].shift(1))).shift(atraso)
    na_regressao = mes >= mes.iloc[0] + (defasagens + 1)
    antes, choque = na_regressao & (mes <= "2020-02"), (mes >= "2020-03") & (mes <= fim)
    alimentos = tabela["ipca_alimentos"]
    aceleracao = alimentos[choque].mean() - alimentos[antes].mean()
    return {
        "aceleracao_todos_os_meses": alimentos[choque].mean() - alimentos[mes <= "2020-02"].mean(),
        "aceleracao": aceleracao,
        "fracao": (contribuicao[choque].mean() - contribuicao[antes].mean()) / aceleracao,
    }


def _mascaras(indice: pd.PeriodIndex, fim: str = "2022-12") -> list[np.ndarray]:
    """Antes, choque e depois, escritos sem o módulo."""
    inicio, fim = pd.Period("2020-03", freq="M"), pd.Period(fim, freq="M")
    return [np.asarray(indice < inicio), np.asarray((indice >= inicio) & (indice <= fim)), np.asarray(indice > fim)]


def _fracoes_por_minimos_quadrados(x_do_ajuste, y_do_ajuste, niveis, x_da_conta, antes, choque, aceleracao):
    """[simples, controlada]: coeficientes tirados de umas linhas, fração medida em outras (ou nas mesmas)."""
    fracoes = []
    for colunas in ([], niveis):
        matriz = np.column_stack([np.ones(len(y_do_ajuste)), *colunas, x_do_ajuste])
        betas = np.linalg.lstsq(matriz, y_do_ajuste, rcond=None)[0][1 + len(colunas) :]
        contribuicao = x_da_conta @ betas
        fracoes.append((contribuicao[choque].mean() - contribuicao[antes].mean()) / aceleracao)
    return fracoes


def test_fracao_acompanhada_quando_metade_do_choque_vem_do_canal():
    # no choque o diesel sobe 4% a mais por mês (0,5 x 4 = +2 pontos nos alimentos) e há +2 pontos
    # que nenhum canal explica: metade da aceleração passa pelos canais, toda ela pelo diesel
    tabela = _mensal(diesel_no_choque=4.0, extra_no_choque=2.0)
    # os três primeiros meses não entram na regressão com duas defasagens. Com inflação bem mais alta
    # neles, a aceleração de todos os meses e a dos meses da regressão ficam diferentes de propósito.
    tabela.loc[:2, "ipca_alimentos"] += 5.0
    aceleracao = eventos.canais(tabela, defasagens=2, n_boot=100)["aceleracao"]
    assert aceleracao["de"] == "antes da COVID (até fev/2020)"
    assert aceleracao["para"] == "COVID e Ucrânia (mar/2020 a dez/2022)"
    esperado = _a_mao(tabela)
    assert esperado["fracao"] == pytest.approx(0.5, abs=0.15)  # o dado foi construído assim
    assert esperado["aceleracao"] - esperado["aceleracao_todos_os_meses"] > 0.15  # perto de 15/38
    assert aceleracao["alimentos_pontos_ao_mes"] == pytest.approx(esperado["aceleracao_todos_os_meses"], abs=0.006)
    na_regressao = aceleracao["alimentos_pontos_ao_mes_nos_meses_da_regressao"]
    assert na_regressao == pytest.approx(esperado["aceleracao"], abs=0.006)

    controlada, simples = aceleracao[CONTROLADA], aceleracao[SIMPLES]
    assert controlada["fracao_acompanhada"] == pytest.approx(esperado["fracao"], abs=0.02)
    assert controlada["salto_sem_explicacao_pontos_ao_mes"] == pytest.approx(2.0, abs=0.05)
    # a conta fecha: o que os canais acompanham mais o salto sem explicação dá a aceleração
    fechado = controlada["fracao_acompanhada"] * na_regressao + controlada["salto_sem_explicacao_pontos_ao_mes"]
    assert fechado == pytest.approx(na_regressao, abs=0.04)
    assert controlada["por_canal"][DIESEL] == pytest.approx(controlada["fracao_acompanhada"], abs=0.02)
    assert abs(controlada["por_canal"][MUNDO]) < 0.02
    # sem o nível de cada período, a regressão dá ao diesel parte do salto que não é dele
    assert simples["fracao_acompanhada"] > controlada["fracao_acompanhada"] + 0.1
    assert "salto_sem_explicacao_pontos_ao_mes" not in simples
    for estimativa in (simples, controlada):
        assert sum(estimativa["por_canal"].values()) == pytest.approx(estimativa["fracao_acompanhada"], abs=0.02)
        for chave in ("ic90_da_conta_inteira", "ic90_dos_coeficientes"):
            inferior, superior = estimativa[chave]
            assert inferior <= superior


def test_o_nivel_do_terceiro_periodo_tambem_e_controlado():
    # de 2023 em diante os alimentos sobem 2 pontos a mais sem canal nenhum e o diesel, 3% a mais por
    # mês. Se só o período do choque tivesse o seu nível, esse degrau entraria nos coeficientes e na
    # base de comparação, e o salto do choque deixaria de ser o 2,0 com que o dado foi construído.
    tabela = _mensal(extra_no_choque=2.0)
    depois = (tabela["mes"] >= "2023-01-01").to_numpy()
    variacao = np.diff(np.log(tabela["diesel_s10"].to_numpy()), prepend=np.log(6.0)) * 100 + 3.0 * depois
    tabela["diesel_s10"] = 6 * np.exp(np.cumsum(variacao) / 100)
    tabela["ipca_alimentos"] += 2.0 * depois + 0.5 * 3.0 * np.concatenate([[False], depois[:-1]])
    controlada = eventos.canais(tabela, defasagens=2, n_boot=20)["aceleracao"][CONTROLADA]
    assert controlada["salto_sem_explicacao_pontos_ao_mes"] == pytest.approx(2.0, abs=0.05)
    assert controlada["fracao_acompanhada"] == pytest.approx(_a_mao(tabela)["fracao"], abs=0.02)


def test_por_canal_separa_os_canais_com_efeito_em_mais_de_uma_defasagem():
    # diesel: 0,5 x o mês anterior (já no dado) + 0,2 x dois meses antes, com +4% por mês no choque
    # mundo: 0,4 x o próprio mês + 0,3 x dois meses antes, com +3% por mês no choque
    tabela = _mensal(diesel_no_choque=4.0)
    choque = ((tabela["mes"] >= "2020-03-01") & (tabela["mes"] <= "2022-12-01")).to_numpy()
    var_diesel = 100 * np.diff(np.log(tabela["diesel_s10"].to_numpy()), prepend=np.log(6.0))
    var_mundo = 100 * np.diff(np.log(tabela["alimentos_mundo_reais"].to_numpy()), prepend=np.log(500.0)) + 3.0 * choque
    tabela["alimentos_mundo_reais"] = 500 * np.exp(np.cumsum(var_mundo) / 100)

    def atrasada(variacao, meses):
        return np.concatenate([np.zeros(meses), variacao[:-meses]])

    do_diesel = 0.5 * atrasada(var_diesel, 1) + 0.2 * atrasada(var_diesel, 2)
    do_mundo = 0.4 * var_mundo + 0.3 * atrasada(var_mundo, 2)
    tabela["ipca_alimentos"] += 0.2 * atrasada(var_diesel, 2) + do_mundo

    antes = (np.arange(len(tabela)) >= 3) & (tabela["mes"] <= "2020-02-01").to_numpy()  # 3 meses fora da regressão
    alimentos = tabela["ipca_alimentos"].to_numpy()
    aceleracao = alimentos[choque].mean() - alimentos[antes].mean()
    esperado = {
        DIESEL: (do_diesel[choque].mean() - do_diesel[antes].mean()) / aceleracao,
        MUNDO: (do_mundo[choque].mean() - do_mundo[antes].mean()) / aceleracao,
    }
    assert 0.3 < esperado[DIESEL] < 0.8  # os dois canais pesam: perto de 0,7 x 4 de 0,7 x 4 + 0,7 x 3
    assert esperado[MUNDO] == pytest.approx(1 - esperado[DIESEL], abs=0.01)  # não há componente por fora
    resultado = eventos.canais(tabela, defasagens=2, n_boot=20)["aceleracao"]
    for nome in (SIMPLES, CONTROLADA):
        assert resultado[nome]["por_canal"] == pytest.approx(esperado, abs=0.02)


def test_reamostra_tem_o_tamanho_da_amostra_e_e_feita_de_blocos_seguidos():
    rng = np.random.default_rng(0)
    for _ in range(50):
        linhas = eventos._reamostra(rng, 20, 6)
        assert len(linhas) == 20
        assert linhas.min() >= 0 and linhas.max() <= 19
        blocos = linhas[:18].reshape(3, 6)
        assert (np.diff(blocos, axis=1) == 1).all()  # seis meses seguidos em cada bloco
        assert blocos[:, 0].max() <= 14  # o bloco cabe inteiro na amostra
        assert linhas[19] == linhas[18] + 1  # o último bloco é cortado para fechar a conta
    inicios = {int(eventos._reamostra(rng, 20, 6)[0]) for _ in range(300)}
    assert inicios == set(range(15))  # qualquer mês em que o bloco caiba pode ser o primeiro
    # um bloco do tamanho da amostra, ou maior, só pode ser a amostra inteira
    assert eventos._reamostra(rng, 20, 20).tolist() == list(range(20))
    assert eventos._reamostra(rng, 20, 500).tolist() == list(range(20))


def test_os_dois_intervalos_sao_percentis_de_reamostras_refeitas_a_mao():
    tabela = _mensal(ruido=1.0, diesel_no_choque=4.0, extra_no_choque=2.0)
    aceleracao = eventos.canais(tabela, defasagens=2, n_boot=200, bloco=6, semente=42)["aceleracao"]
    ajuste = eventos._regressao(eventos.preparar(tabela), 2)
    principal = eventos._estimativas(ajuste, eventos.PERIODOS)
    x, y = ajuste["x"], ajuste["y"]
    antes, choque, depois = _mascaras(ajuste["indice"])
    observada = y[choque].mean() - y[antes].mean()

    # só os coeficientes: reamostra a amostra toda, reajusta, e mede a fração nos canais observados
    coeficientes = eventos._sorteios_dos_coeficientes(ajuste, principal, n_boot=200, bloco=6, semente=42)
    rng = np.random.default_rng(42)
    for sorteio in range(3):
        linhas = eventos._reamostra(rng, len(y), 6)
        niveis = [choque[linhas].astype(float), depois[linhas].astype(float)]
        a_mao = _fracoes_por_minimos_quadrados(x[linhas], y[linhas], niveis, x, antes, choque, observada)
        assert coeficientes[sorteio] == pytest.approx(a_mao)

    # a conta inteira: reamostra dentro de cada período e refaz coeficientes, canais e aceleração
    inteira = eventos._sorteios_da_conta_inteira(ajuste, principal, n_boot=200, bloco=6, semente=42)
    rng = np.random.default_rng(42)
    posicoes = [np.flatnonzero(mascara) for mascara in (antes, choque, depois)]
    for sorteio in range(3):
        linhas = np.concatenate([p[eventos._reamostra(rng, len(p), 6)] for p in posicoes])
        assert [mascara[linhas].sum() for mascara in (antes, choque, depois)] == [len(p) for p in posicoes]
        a, c, d = antes[linhas], choque[linhas], depois[linhas]
        sorteada = y[linhas][c].mean() - y[linhas][a].mean()
        niveis = [c.astype(float), d.astype(float)]
        a_mao = _fracoes_por_minimos_quadrados(x[linhas], y[linhas], niveis, x[linhas], a, c, sorteada)
        assert inteira[sorteio] == pytest.approx(a_mao)

    for sorteios, chave in ((inteira, "ic90_da_conta_inteira"), (coeficientes, "ic90_dos_coeficientes")):
        assert sorteios.shape == (200, 2)
        limites = np.percentile(sorteios, [5, 95], axis=0)
        for coluna, nome in enumerate((SIMPLES, CONTROLADA)):
            assert aceleracao[nome][chave] == [round(float(v), 2) for v in limites[:, coluna]]
            inferior, superior = aceleracao[nome][chave]
            assert superior - inferior > 0.05  # com ruído, a conta muda de uma reamostra para a outra
    assert aceleracao[CONTROLADA]["ic90_da_conta_inteira"] != aceleracao[CONTROLADA]["ic90_dos_coeficientes"]

    # com blocos do tamanho da amostra não há o que sortear: toda reamostra é a própria amostra
    pontos = [principal["fracoes"][SIMPLES], principal["fracoes"][CONTROLADA]]
    for sorteios in (eventos._sorteios_dos_coeficientes, eventos._sorteios_da_conta_inteira):
        assert sorteios(ajuste, principal, n_boot=4, bloco=500, semente=1) == pytest.approx(np.tile(pontos, (4, 1)))


def test_canais_usa_os_padroes_publicados_e_repassa_semente_e_bloco():
    padroes = {nome: p.default for nome, p in inspect.signature(eventos.canais).parameters.items() if nome != "mensal"}
    assert padroes == {"defasagens": 4, "n_boot": 4000, "bloco": 6, "semente": 42}
    tabela = _mensal(ruido=1.0, diesel_no_choque=4.0, extra_no_choque=2.0)
    assert eventos.canais(tabela, n_boot=20)["defasagens"] == 4

    def intervalos(**parametros):
        c = eventos.canais(tabela, defasagens=2, n_boot=60, **parametros)["aceleracao"][CONTROLADA]
        return c["ic90_da_conta_inteira"], c["ic90_dos_coeficientes"], c["fracao_acompanhada"]

    assert intervalos(semente=1) == intervalos(semente=1)
    for de_uma, de_outra in zip(intervalos(semente=1)[:2], intervalos(semente=2)[:2], strict=True):
        assert de_uma != de_outra
    inteira, coeficientes, ponto = intervalos(bloco=500)
    assert inteira == coeficientes == [ponto, ponto]
    for invalido in ({"n_boot": 0}, {"bloco": 0}):
        with pytest.raises(ValueError, match="n_boot e bloco"):
            eventos.canais(tabela, **invalido)


def test_quando_tudo_vem_do_canal_a_fracao_e_um_em_toda_a_grade():
    # sem componente por fora, os alimentos são intercepto + 0,5 x diesel do mês anterior: os canais
    # acompanham toda a aceleração com qualquer número de defasagens e qualquer data de corte
    aceleracao = eventos.canais(_mensal(diesel_no_choque=6.0), defasagens=2, n_boot=20)["aceleracao"]
    grade = aceleracao["por_defasagens_e_fim_do_choque"]
    assert list(grade) == ["1", "2", "3", "4", "6", "8"]
    for linha in grade.values():
        assert tuple(linha) == CORTES
        for celula in linha.values():
            assert celula[SIMPLES] == pytest.approx(1.0, abs=0.03)
            assert celula[CONTROLADA] == pytest.approx(1.0, abs=0.03)
    assert aceleracao[CONTROLADA]["salto_sem_explicacao_pontos_ao_mes"] == pytest.approx(0.0, abs=0.02)
    for nome in (SIMPLES, CONTROLADA):
        faixa = aceleracao["faixa_das_especificacoes"][nome]
        assert faixa["combinacoes"] == faixa["com_metade_ou_mais"] == 18
        assert faixa["minimo"] == pytest.approx(1.0, abs=0.03)


def test_grade_muda_com_o_numero_de_defasagens_e_com_o_controle():
    # o diesel leva três meses para chegar aos alimentos: com uma ou duas defasagens a regressão não
    # o enxerga. A controlada fica longe de 1; a simples fica bem acima dela, porque o diesel dos
    # meses que ela vê sobe no mesmo período e ela lhe atribui parte da alta.
    tabela = _mensal(diesel_no_choque=6.0, atraso=3)
    aceleracao = eventos.canais(tabela, defasagens=4, n_boot=20)["aceleracao"]
    grade = aceleracao["por_defasagens_e_fim_do_choque"]
    for k in ("1", "2"):
        assert grade[k]["2022-12"][CONTROLADA] < 0.5
        assert grade[k]["2022-12"][SIMPLES] > grade[k]["2022-12"][CONTROLADA] + 0.2
    for k in ("3", "4", "6", "8"):
        assert grade[k]["2022-12"][CONTROLADA] == pytest.approx(1.0, abs=0.03)
        assert grade[k]["2022-12"][SIMPLES] == pytest.approx(1.0, abs=0.03)
    for nome in (SIMPLES, CONTROLADA):
        # a especificação principal é uma das células da grade
        assert grade["4"]["2022-12"][nome] == aceleracao[nome]["fracao_acompanhada"]
        valores = [celula[nome] for linha in grade.values() for celula in linha.values()]
        faixa = aceleracao["faixa_das_especificacoes"][nome]
        assert (faixa["minimo"], faixa["maximo"], faixa["combinacoes"]) == (min(valores), max(valores), 18)
    assert 12 <= aceleracao["faixa_das_especificacoes"][CONTROLADA]["com_metade_ou_mais"] < 18


def test_grade_muda_com_o_mes_em_que_o_choque_termina():
    antes, choque, depois = list(eventos.PERIODOS)
    assert eventos._periodos_com_fim("2022-06") == {
        antes: (None, "2020-02"),
        choque: ("2020-03", "2022-06"),
        depois: ("2022-07", None),
    }
    assert eventos._periodos_com_fim("2023-06")[choque] == ("2020-03", "2023-06")
    assert eventos._periodos_com_fim("2023-06")[depois] == ("2023-07", None)
    assert eventos._periodos_com_fim("2022-12") == eventos.PERIODOS
    assert eventos.SENSIBILIDADE_FIM_DO_CHOQUE == CORTES

    # neste dado o choque (diesel e componente sem canal) acaba em jun/2022
    tabela = _mensal(diesel_no_choque=4.0, extra_no_choque=2.0, fim_do_choque="2022-06-01")
    grade = eventos.canais(tabela, defasagens=2, n_boot=20)["aceleracao"]["por_defasagens_e_fim_do_choque"]
    # com o corte no lugar, a conta é a feita à mão
    assert grade["2"]["2022-06"][CONTROLADA] == pytest.approx(_a_mao(tabela, fim="2022-06")["fracao"], abs=0.02)
    # com o corte depois da hora, o fim do choque fica dentro do período: diesel e componente sem
    # canal caem juntos, e a regressão dá ao diesel parte do que não é dele. Quanto mais tarde, mais.
    assert grade["2"]["2022-12"][CONTROLADA] > grade["2"]["2022-06"][CONTROLADA] + 0.03
    assert grade["2"]["2023-06"][CONTROLADA] > grade["2"]["2022-12"][CONTROLADA]
    # cada célula é a conta com aquelas defasagens e aquele corte, e não a principal repetida
    dados = eventos.preparar(tabela)
    for k in eventos.SENSIBILIDADE_DEFASAGENS:
        for fim in CORTES:
            ajuste = eventos._regressao(dados, k)
            x, y = ajuste["x"], ajuste["y"]
            a, c, d = _mascaras(ajuste["indice"], fim)
            niveis = [c.astype(float), d.astype(float)]
            a_mao = _fracoes_por_minimos_quadrados(x, y, niveis, x, a, c, y[c].mean() - y[a].mean())
            assert grade[str(k)][fim] == {SIMPLES: round(a_mao[0], 2), CONTROLADA: round(a_mao[1], 2)}


def test_resumo_da_grade_usa_os_valores_sem_arredondar(monkeypatch):
    exata = {
        "1": {"2022-06": {SIMPLES: 0.496, CONTROLADA: 0.2119}, "2022-12": {SIMPLES: 0.5, CONTROLADA: 0.1}},
        "2": {"2022-06": {SIMPLES: 0.9, CONTROLADA: 0.2228}, "2022-12": {SIMPLES: 0.7, CONTROLADA: 0.4}},
        "3": {"2022-06": None, "2022-12": None},
    }
    # os dois valores do meio são 0,2119 e 0,2228, e a mediana é 0,217; com eles já arredondados
    # (0,21 e 0,22) seria 0,215, que vira 0,21. A média dos quatro seria 0,23.
    assert eventos._faixa(exata, CONTROLADA) == {
        "combinacoes": 4,
        "minimo": 0.1,
        "mediana": 0.22,
        "maximo": 0.4,
        "com_metade_ou_mais": 0,
    }
    # 0,496 aparece como 0,5 na tabela, mas não chega à metade; 0,5 exato chega
    assert eventos._faixa(exata, SIMPLES) == {
        "combinacoes": 4,
        "minimo": 0.5,
        "mediana": 0.6,
        "maximo": 0.9,
        "com_metade_ou_mais": 3,
    }
    assert eventos._faixa({"1": {"2022-06": None}}, SIMPLES) is None

    # no relatório a grade sai arredondada, e o resumo, da grade exata
    monkeypatch.setattr(eventos, "_grade", lambda *_: exata)
    aceleracao = eventos.canais(_mensal(diesel_no_choque=6.0), defasagens=2, n_boot=20)["aceleracao"]
    assert aceleracao["por_defasagens_e_fim_do_choque"] == {
        "1": {"2022-06": {SIMPLES: 0.5, CONTROLADA: 0.21}, "2022-12": {SIMPLES: 0.5, CONTROLADA: 0.1}},
        "2": {"2022-06": {SIMPLES: 0.9, CONTROLADA: 0.22}, "2022-12": {SIMPLES: 0.7, CONTROLADA: 0.4}},
        "3": {"2022-06": None, "2022-12": None},
    }
    assert aceleracao["faixa_das_especificacoes"][CONTROLADA]["mediana"] == 0.22
    assert aceleracao["faixa_das_especificacoes"][SIMPLES]["com_metade_ou_mais"] == 3


def test_grade_inclui_as_defasagens_da_conta_principal_e_nao_repete_cortes():
    # cinco defasagens não estão na lista de sensibilidade: entram na grade assim mesmo
    aceleracao = eventos.canais(_mensal(diesel_no_choque=6.0), defasagens=5, n_boot=20)["aceleracao"]
    grade = aceleracao["por_defasagens_e_fim_do_choque"]
    assert list(grade) == ["1", "2", "3", "4", "5", "6", "8"]
    assert grade["5"]["2022-12"] == {nome: aceleracao[nome]["fracao_acompanhada"] for nome in (SIMPLES, CONTROLADA)}
    assert aceleracao["faixa_das_especificacoes"][CONTROLADA]["combinacoes"] == 21

    # tabela até out/2022: o corte de dez/2022 é "até o fim da tabela", e o de jun/2023 seria a mesma conta
    # tabela até dez/2022: o corte de jun/2023 repetiria o de dez/2022
    for meses in (70, 72):
        curta = eventos.canais(_mensal(n=meses, diesel_no_choque=6.0), defasagens=2, n_boot=20)["aceleracao"]
        grade = curta["por_defasagens_e_fim_do_choque"]
        for linha in grade.values():
            assert linha["2022-06"] is not None and linha["2022-12"] is not None
            assert linha["2023-06"] is None
        assert grade["2"]["2022-12"][CONTROLADA] == curta[CONTROLADA]["fracao_acompanhada"]
        assert curta["faixa_das_especificacoes"][CONTROLADA]["combinacoes"] == 12
    # tabela até jun/2021: os três cortes dariam a mesma conta, e só o primeiro fica
    curtissima = eventos.canais(_mensal(n=54, diesel_no_choque=6.0), defasagens=2, n_boot=20)["aceleracao"]
    for linha in curtissima["por_defasagens_e_fim_do_choque"].values():
        assert linha["2022-06"] is not None and linha["2022-12"] is None and linha["2023-06"] is None
    assert curtissima["por_defasagens_e_fim_do_choque"]["2"]["2022-06"][CONTROLADA] == (
        curtissima[CONTROLADA]["fracao_acompanhada"]
    )


def test_sem_meses_para_a_conta_nao_ha_numero():
    def aceleracao(**dado):
        return eventos.canais(_mensal(diesel_no_choque=6.0, **dado), defasagens=2, n_boot=20)["aceleracao"]

    assert aceleracao(n=30) is None  # termina em jun/2019: não chega ao período do choque
    assert aceleracao(n=45) is None  # até set/2020 são sete meses de choque
    assert aceleracao(n=49) is None  # onze meses
    assert aceleracao(n=50) is not None  # doze meses: o mínimo
    assert eventos.MIN_MESES_POR_PERIODO == 12
    # começando em jul/2018, oito defasagens deixam 11 meses antes da COVID: essa linha fica sem conta
    tardia = aceleracao(n=78, inicio="2018-07-01")
    grade = tardia["por_defasagens_e_fim_do_choque"]
    assert all(celula is None for celula in grade["8"].values())
    assert all(celula is not None for k in ("1", "2", "3", "4", "6") for celula in grade[k].values())
    assert tardia["faixa_das_especificacoes"][CONTROLADA]["combinacoes"] == 15


def test_sem_aceleracao_clara_ou_com_coeficientes_demais_nao_ha_fracao():
    # 24 meses antes da COVID e 24 de choque, com a inflação alternando +1 e -1 em volta do nível
    indice = pd.period_range("2018-03", periods=48, freq="M")
    assert [mascara.sum() for mascara in _mascaras(indice)] == [24, 24, 0]
    rng = np.random.default_rng(0)
    sobe_e_desce, no_choque = np.tile([1.0, -1.0], 24), np.arange(48) >= 24
    erro = np.sqrt(2 * (24 / 23) / 24)  # a variância amostral de cada período é 24/23
    canais = rng.normal(size=(48, 2))
    # uma aceleração que não passa de dois erros-padrão é ruído, e dividir por ela dá qualquer coisa;
    # desaceleração e aceleração nula também ficam sem fração
    for salto, ha_conta in ((2.1 * erro, True), (1.9 * erro, False), (0.0, False), (-3.0, False)):
        ajuste = {"x": canais, "y": sobe_e_desce + salto * no_choque, "indice": indice}
        assert (eventos._estimativas(ajuste, eventos.PERIODOS) is not None) == ha_conta
    # dois meses por coeficiente: intercepto + um nível + 22 canais são 24 coeficientes para 48 meses
    for colunas, ha_conta in ((22, True), (23, False)):
        ajuste = {"x": rng.normal(size=(48, colunas)), "y": sobe_e_desce + 5.0 * no_choque, "indice": indice}
        assert (eventos._estimativas(ajuste, eventos.PERIODOS) is not None) == ha_conta
    # o dado sintético de sempre, com os alimentos desacelerando no choque
    assert eventos.canais(_mensal(extra_no_choque=-3.0), defasagens=2, n_boot=20)["aceleracao"] is None
