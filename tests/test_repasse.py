"""Repasse do barril à bomba e cenários, em pandas (sem Spark)."""

import numpy as np
import pandas as pd
import pytest

from anp_precos import repasse
from anp_precos.esquema import LITROS_POR_BARRIL

DOLAR = 5.0


def _semanal(n=320, betas=(0.0, 0.10, 0.10, 0.05), ruido=0.05, semente=1, buraco=None):
    """Série sintética: a bomba repassa `betas` do barril ao longo das semanas seguintes."""
    rng = np.random.default_rng(semente)
    var_barril = rng.normal(0, 4.0, n)  # % por semana, em log
    var_bomba = np.zeros(n)
    for k, beta in enumerate(betas):
        var_bomba[k:] += beta * var_barril[: n - k]
    var_bomba += rng.normal(0, ruido, n)
    brent = 80 * np.exp(np.cumsum(var_barril) / 100)
    tabela = pd.DataFrame(
        {
            "semana": pd.date_range("2018-01-01", periods=n, freq="7D"),
            "brent_usd": brent,
            "dolar": DOLAR,
            "brent_reais_litro": brent * DOLAR / LITROS_POR_BARRIL,
            "gasolina": 5 * np.exp(np.cumsum(var_bomba) / 100),
        }
    )
    if buraco:
        tabela = tabela.drop(index=range(*buraco)).reset_index(drop=True)
    return tabela


def test_curva_recupera_o_repasse_e_o_intervalo_e_de_90_por_cento():
    curva = repasse.curva(_semanal(), "gasolina", defasagens=6, n_boot=300)
    verdade = np.cumsum([0.0, 0.10, 0.10, 0.05, 0, 0, 0])
    assert curva["semanas"] > 300
    assert curva["acumulado"] == pytest.approx(verdade, abs=0.01)
    assert all(inf <= v <= sup for inf, v, sup in zip(curva["ic90_inf"], verdade, curva["ic90_sup"]))
    # o intervalo publicado é exatamente o percentil 5 a 95 dos sorteios (não 25 a 75, não mínimo a máximo)
    acumulados = np.cumsum(curva["_betas_sorteados"], axis=1)
    assert curva["ic90_inf"] == pytest.approx(np.percentile(acumulados, 5, axis=0), abs=1e-4)
    assert curva["ic90_sup"] == pytest.approx(np.percentile(acumulados, 95, axis=0), abs=1e-4)
    assert curva["_betas_sorteados"].shape == (300, 7)


def test_semana_faltante_nao_e_emendada_com_a_vizinha():
    # 10 semanas sem dado no meio (como a pesquisa da ANP em 2020): as variações que tocam o buraco somem
    inteira = repasse.curva(_semanal(), "gasolina", defasagens=4, n_boot=50)
    furada = repasse.curva(_semanal(buraco=(150, 160)), "gasolina", defasagens=4, n_boot=50)
    assert furada["semanas"] == inteira["semanas"] - 10 - 5  # 10 do buraco + 1 de retorno + 4 defasagens
    assert furada["acumulado"] == pytest.approx(inteira["acumulado"], abs=0.02)


def test_recorte_por_periodo_nao_conta_a_semana_do_corte_duas_vezes():
    tabela = _semanal()
    corte = str(tabela["semana"].iloc[200].date())
    inteira = repasse.curva(tabela, "gasolina", defasagens=4, n_boot=20)
    antes = repasse.curva(tabela, "gasolina", fim=corte, defasagens=4, n_boot=20)
    depois = repasse.curva(tabela, "gasolina", inicio=corte, defasagens=4, n_boot=20)
    assert depois["periodo"][0] == corte  # a primeira semana depois do corte já entra (defasagens vêm de antes)
    assert depois["semanas"] == len(tabela) - 200
    assert antes["periodo"][1] == str(tabela["semana"].iloc[199].date())  # o fim é exclusivo
    assert antes["semanas"] + depois["semanas"] == inteira["semanas"]


def test_poucas_semanas_falha_com_mensagem():
    with pytest.raises(ValueError, match="Poucas semanas"):
        repasse.curva(_semanal(n=30), "gasolina", defasagens=16, n_boot=10)


def _curva_fixa(betas, sorteios=None) -> dict:
    betas = np.array(betas, dtype=float)
    return {"produto": "gasolina", "_betas": betas, "_betas_sorteados": np.array(sorteios or [betas])}


def _plana(n=40, brent=80.0, bomba=6.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "semana": pd.date_range("2026-01-05", periods=n, freq="7D"),
            "brent_usd": brent,
            "dolar": DOLAR,
            "brent_reais_litro": brent * DOLAR / LITROS_POR_BARRIL,
            "gasolina": bomba,
        }
    )


def test_k_semanas_depois_do_salto_usa_o_repasse_acumulado_ate_k():
    # barril parado até hoje; a hipótese dobra o barril. Repasse: 10% na semana do salto, +20% na seguinte.
    trajetoria = repasse.simular(_plana(), _curva_fixa([0.10, 0.20, 0.0]), {"brent_usd": 160.0, "dolar": DOLAR}, 3)
    salto = 100 * np.log(2)
    acumulado = np.array([0.10, 0.30, 0.30, 0.30])  # o mesmo vetor da curva de repasse
    assert trajetoria["semanas_depois"].tolist() == [0, 1, 2, 3]
    assert trajetoria["preco"].to_numpy() == pytest.approx(6.0 * np.exp(acumulado * salto / 100))
    assert trajetoria["variacao_pct"].iloc[1] == pytest.approx(100 * (2**0.30 - 1))


def test_barril_parado_e_sem_alta_recente_nao_mexe_no_preco():
    trajetoria = repasse.simular(_plana(), _curva_fixa([0.1, 0.2, 0.3]), {"brent_usd": 80.0, "dolar": DOLAR}, 6)
    assert trajetoria["variacao_pct"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_repasse_pendente_de_uma_alta_que_ja_aconteceu():
    # o barril subiu 10% na última semana observada; com repasse só na defasagem 2, o efeito
    # ainda não chegou à bomba e aparece mesmo no cenário "tudo fica onde está"
    tabela = _plana()
    tabela.loc[tabela.index[-1], ["brent_usd", "brent_reais_litro"]] *= 1.10
    trajetoria = repasse.simular(tabela, _curva_fixa([0.0, 0.0, 0.5]), {"brent_usd": 88.0, "dolar": DOLAR}, 3)
    alta = 100 * np.log(1.10)
    devido = 100 * (np.exp(0.5 * alta / 100) - 1)
    # semana do salto (0): a alta passada está na defasagem 1, ainda nada; uma semana depois, chega inteira
    assert trajetoria["variacao_pct"].tolist() == pytest.approx([0.0, devido, devido, devido], abs=1e-9)


def test_buraco_curto_no_barril_nao_apaga_o_repasse_pendente():
    # o barril subiu 50% numa semana que ficou sem cotação (feriado): o pendente não pode virar zero
    tabela = _plana()
    tabela.loc[tabela.index[-2:], ["brent_usd", "brent_reais_litro"]] *= 1.5
    com_buraco = tabela.copy()
    com_buraco.loc[com_buraco.index[-2], ["brent_usd", "brent_reais_litro"]] = np.nan
    hoje = {"brent_usd": 120.0, "dolar": DOLAR}
    curva = _curva_fixa([0.0, 0.0, 0.0, 0.5])
    inteira = repasse.simular(tabela, curva, hoje, 4)["variacao_pct"]
    furada = repasse.simular(com_buraco, curva, hoje, 4)["variacao_pct"]
    assert inteira.iloc[-1] == pytest.approx(100 * (1.5**0.5 - 1))
    assert furada.iloc[-1] == pytest.approx(inteira.iloc[-1])  # a alta continua devida, só muda a semana em que cai


def test_buraco_longo_no_barril_impede_o_cenario():
    tabela = _plana()
    tabela.loc[tabela.index[-6:-1], ["brent_usd", "brent_reais_litro"]] = np.nan
    with pytest.raises(ValueError, match="buraco de mais de 2 semanas"):
        repasse.simular(tabela, _curva_fixa([0.1] * 9), {"brent_usd": 80.0, "dolar": DOLAR}, 4)


def test_intervalo_e_o_percentil_5_a_95_e_a_queda_do_barril_inverte_as_pontas():
    sorteios = [[b] for b in np.linspace(0.0, 1.0, 101)]  # percentil 5 = 0,05; percentil 95 = 0,95
    curva = _curva_fixa([0.5], sorteios=sorteios)
    alta = repasse.simular(_plana(), curva, {"brent_usd": 160.0, "dolar": DOLAR}, 0).iloc[0]
    assert alta["preco"] == pytest.approx(6.0 * 2**0.5)
    assert alta["preco_ic90_inf"] == pytest.approx(6.0 * 2**0.05)
    assert alta["preco_ic90_sup"] == pytest.approx(6.0 * 2**0.95)
    queda = repasse.simular(_plana(), curva, {"brent_usd": 40.0, "dolar": DOLAR}, 0).iloc[0]
    assert queda["preco_ic90_inf"] == pytest.approx(6.0 * 0.5**0.95)  # na queda, o repasse maior dá o preço menor
    assert queda["preco_ic90_sup"] == pytest.approx(6.0 * 0.5**0.05)
    assert queda["preco_ic90_inf"] < queda["preco"] < queda["preco_ic90_sup"] < 6.0


def test_hipoteses_saem_das_semanas_completas_da_propria_serie():
    tabela = _plana(n=80)
    tabela.loc[10, "brent_usd"] = 200.0  # pico antigo, fora da janela de 52 semanas
    tabela.loc[50, "brent_usd"] = 120.0
    tabela.loc[60, "brent_usd"] = 60.0
    tabela.loc[5, "dolar"] = 6.5
    # semanas sem barril em reais (fonte incompleta) não podem virar pico, piso nem dólar máximo
    tabela.loc[70, ["brent_usd", "dolar", "brent_reais_litro"]] = [300.0, 9.0, np.nan]
    tabela.loc[[78, 79], "brent_reais_litro"] = np.nan
    tabela.loc[[78, 79], "brent_usd"] = 10.0
    hipoteses = repasse.hipoteses_do_historico(tabela, janela=52)
    assert hipoteses["Barril e dólar ficam onde estão"] == {"brent_usd": 80.0, "dolar": DOLAR}
    assert hipoteses["Escalada: Brent volta ao pico de 12 meses"]["brent_usd"] == 120.0
    assert hipoteses["Alívio: Brent volta ao piso de 12 meses"]["brent_usd"] == 60.0
    assert hipoteses["Escalada com dólar no máximo da série"] == {"brent_usd": 120.0, "dolar": 6.5}


def test_ponto_de_partida_e_a_ultima_semana_com_barril_e_bomba():
    tabela = _plana(n=10)
    tabela["dias_dolar"] = 5
    # a última semana ainda não tem barril na fonte (só 2 cotações de dólar), mas a bomba já saiu
    tabela.loc[9, ["brent_usd", "brent_reais_litro", "dolar", "dias_dolar"]] = [np.nan, np.nan, 5.9, 2]
    tabela.loc[9, "gasolina"] = 9.99
    tabela.loc[8, "dolar"] = 5.5  # semana completa mais recente
    atual = repasse.situacao_atual(tabela, ["gasolina"])
    assert atual["semana"] == str(tabela["semana"].iloc[8].date())
    assert atual["bomba"]["gasolina"] == 6.0
    assert atual["dolar_mais_recente"] == {"semana": atual["semana"], "dolar": 5.5}  # a de 2 cotações não conta
    # e o cenário parte do mesmo preço, não do que saiu depois
    trajetoria = repasse.simular(tabela, _curva_fixa([0.0]), {"brent_usd": 80.0, "dolar": DOLAR}, 0)
    assert trajetoria["preco"].iloc[0] == pytest.approx(6.0)


def test_bomba_muito_atras_do_barril_impede_o_cenario():
    tabela = _plana(n=20)
    tabela.loc[tabela.index[-3:], "gasolina"] = np.nan  # 3 semanas sem preço na bomba
    with pytest.raises(ValueError, match="3 semanas antes do barril"):
        repasse.simular(tabela, _curva_fixa([0.5]), {"brent_usd": 120.0, "dolar": DOLAR}, 2)
    tabela.loc[tabela.index[-3], "gasolina"] = 6.0  # 2 semanas de atraso ainda passam
    assert repasse.situacao_atual(tabela, ["gasolina"])["semana"] == str(tabela["semana"].iloc[-3].date())


def test_tabela_de_cenarios_tem_sinal_e_marcos_certos():
    hipoteses = {"alta": {"brent_usd": 120.0, "dolar": DOLAR}, "queda": {"brent_usd": 40.0, "dolar": DOLAR}}
    tabela = repasse.tabela_de_cenarios(_plana(), {"Gasolina": _curva_fixa([0.1, 0.1, 0.1])}, hipoteses, marcos=(0, 2))
    linhas = tabela.set_index(["cenario", "semanas_depois"])
    assert linhas.at[("alta", 2), "barril_reais_var_pct"] == 50.0
    assert linhas.at[("queda", 2), "barril_reais_var_pct"] == -50.0
    assert linhas.at[("alta", 0), "variacao_pct"] == pytest.approx(100 * (1.5**0.1 - 1), abs=0.01)
    assert linhas.at[("alta", 2), "variacao_pct"] == pytest.approx(100 * (1.5**0.3 - 1), abs=0.01)
    assert linhas.at[("queda", 2), "preco"] < 6.0 < linhas.at[("alta", 2), "preco"]
    assert (tabela["preco_atual"] == 6.0).all()


def test_relatorio_monta_curvas_cenarios_e_a_leitura_com_paridade():
    tabela = _semanal(n=520)
    tabela["diesel_s10"] = tabela["gasolina"] * 1.1
    tabela["semana"] = pd.date_range("2017-01-02", periods=len(tabela), freq="7D")
    analise = repasse.relatorio(tabela, n_boot=30)
    assert set(analise["curvas"]) == {"Gasolina comum", "Diesel S10"}
    assert set(analise["curvas"]["Gasolina comum"]) == set(repasse.PERIODOS)
    cenarios = analise["cenarios"]
    assert set(cenarios["curva"]) == {
        "Gasolina comum, repasse de hoje",
        "Diesel S10, repasse de hoje",
        "Gasolina comum, se a paridade voltasse",
    }
    assert set(cenarios["semanas_depois"]) == {4, 8, 16}
    assert (cenarios["preco_ic90_inf"] <= cenarios["preco_ic90_sup"]).all()
    # a leitura "se a paridade voltasse" usa a curva do período anterior ao corte, não a de hoje
    paridade = analise["curvas"]["Gasolina comum"][repasse.ATE_O_CORTE]
    de_hoje = analise["curvas"]["Gasolina comum"][repasse.DEPOIS_DO_CORTE]
    assert paridade["periodo"][1] < "2023-05-15" <= de_hoje["periodo"][0]
    resumo = repasse.resumo_das_curvas(analise["curvas"])
    assert set(resumo["Diesel S10"][repasse.DEPOIS_DO_CORTE]["repasse_acumulado"]) == {"4", "8", "16"}
    assert resumo["Gasolina comum"][repasse.ATE_O_CORTE]["repasse_acumulado"]["16"] == paridade["acumulado"][16]
