import numpy as np
import pandas as pd
import pytest

from anp_precos import modelo


def _tabela(n_semanas=80, municipios=40, semente=0, peso_brent=0.0):
    """`peso_brent` > 0 faz a alta depender também da variação do barril em 4 semanas."""
    rng = np.random.default_rng(semente)
    semanas = pd.date_range("2024-01-01", periods=n_semanas, freq="W-MON")
    linhas = []
    for s in semanas:
        brent = rng.normal(0, 0.05, size=3)  # o barril é o mesmo para todos os municípios da semana
        for m in range(municipios):
            var_1s = rng.normal(0, 0.01)
            sinal = 3 * var_1s + peso_brent * brent[1] + rng.normal(0, 0.01)
            linhas.append(
                {
                    "semana": s,
                    "produto": "GASOLINA" if m % 2 else "ETANOL",
                    "regiao": ["N", "NE", "CO", "SE", "S"][m % 5],
                    "var_1s": var_1s,
                    "var_2s": np.nan if m % 7 == 0 else rng.normal(0, 0.01),
                    "var_4s": rng.normal(0, 0.02),
                    "var_nacional_1s": rng.normal(0, 0.005),
                    "var_nacional_4s": rng.normal(0, 0.01),
                    "dispersao": abs(rng.normal(0.05, 0.02)),
                    "n_postos": int(rng.integers(3, 40)),
                    "premio_ipiranga_pct": np.nan if m % 3 else rng.normal(1, 2),
                    "semana_do_ano": s.isocalendar().week,
                    "var_brent_1s": brent[0],
                    "var_brent_4s": brent[1],
                    "var_brent_8s": brent[2],
                    "sobe_prox_semana": int(sinal > 0.005),
                }
            )
    return pd.DataFrame(linhas)


def test_particao_no_tempo_nao_mistura_semanas_e_tem_embargo():
    p = modelo.particionar_no_tempo(_tabela())
    semana = pd.Timedelta(days=7)
    # 1 semana inteira de folga em cada fronteira (o alvo de t olha t+1)
    assert p.validacao["semana"].min() - p.treino["semana"].max() == 2 * semana
    assert p.teste["semana"].min() - p.validacao["semana"].max() == 2 * semana
    assert p.teste["semana"].nunique() == 26
    assert p.validacao["semana"].nunique() == 25


def test_modelo_aprende_sinal_e_bate_o_acaso():
    resultado = modelo.treinar_e_avaliar(_tabela())
    assert resultado["teste"]["gradient_boosting"]["roc_auc"] > 0.75
    assert set(resultado["teste"]) == {"gradient_boosting", "logistica", "persistencia"}


def test_comparacao_sem_brent_retira_mesmo_as_features_do_barril():
    # aqui o barril carrega sinal de verdade: se a comparação treinasse os dois lados com as
    # mesmas colunas, o ganho seria zero e o teste falharia
    tabela = _tabela(n_semanas=160, peso_brent=0.5)
    resultado = modelo.treinar_e_avaliar(tabela)
    assert set(modelo.NUMERICAS_BRENT) <= set(resultado["_x_teste"].columns)

    comparacao = modelo.comparar_sem_brent(tabela, resultado)
    gb = comparacao["teste"]["gradient_boosting"]
    assert gb["com_brent"] == resultado["teste"]["gradient_boosting"]
    assert gb["com_brent"]["roc_auc"] - gb["sem_brent"]["roc_auc"] > 0.03

    ganho = comparacao["ganho_auc_gradient_boosting"]["teste"]
    assert ganho["semanas"] == 26
    assert ganho["diferenca"] == pytest.approx(gb["com_brent"]["roc_auc"] - gb["sem_brent"]["roc_auc"], abs=2e-4)
    assert ganho["ic90"][0] < ganho["diferenca"] < ganho["ic90"][1]
    assert ganho["ic90"][0] > 0  # com sinal forte, o intervalo por semana não encosta no zero

    por_produto = comparacao["teste_por_produto"]
    assert set(por_produto) == {"ETANOL", "GASOLINA"}
    assert por_produto["GASOLINA"]["roc_auc_com_brent"] > por_produto["GASOLINA"]["roc_auc_sem_brent"]


def test_barril_sem_sinal_nao_mostra_ganho():
    comparacao_base = _tabela()  # peso_brent = 0: o barril é ruído
    resultado = modelo.treinar_e_avaliar(comparacao_base)
    ganho = modelo.comparar_sem_brent(comparacao_base, resultado)["ganho_auc_gradient_boosting"]["teste"]
    assert ganho["ic90"][0] <= 0 <= ganho["ic90"][1] or abs(ganho["diferenca"]) < 0.02


def test_modelo_roda_sem_as_colunas_do_brent_quando_pedido():
    tabela = _tabela().drop(columns=modelo.NUMERICAS_BRENT)
    resultado = modelo.treinar_e_avaliar(tabela, numericas=modelo.NUMERICAS_BASE)
    assert not set(modelo.NUMERICAS_BRENT) & set(resultado["_x_teste"].columns)


def test_tabela_de_features_antiga_falha_com_orientacao():
    tabela = _tabela().drop(columns=modelo.NUMERICAS_BRENT)
    with pytest.raises(ValueError, match="rode o pipeline"):
        modelo.treinar_e_avaliar(tabela)


def test_poucas_semanas_falha_com_mensagem():
    with pytest.raises(ValueError, match="Poucas semanas"):
        modelo.particionar_no_tempo(_tabela(n_semanas=40))
