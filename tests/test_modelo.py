import numpy as np
import pandas as pd
import pytest

from anp_precos import modelo


def _tabela(n_semanas=80, municipios=40, semente=0):
    rng = np.random.default_rng(semente)
    semanas = pd.date_range("2024-01-01", periods=n_semanas, freq="W-MON")
    linhas = []
    for s in semanas:
        for m in range(municipios):
            var_1s = rng.normal(0, 0.01)
            sinal = 3 * var_1s + rng.normal(0, 0.01)
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


def test_poucas_semanas_falha_com_mensagem():
    with pytest.raises(ValueError, match="Poucas semanas"):
        modelo.particionar_no_tempo(_tabela(n_semanas=40))
