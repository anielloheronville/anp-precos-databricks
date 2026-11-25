"""Testes estatísticos sobre o prêmio da Ipiranga (SciPy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def premio_por_produto(mercado: pd.DataFrame, semente: int = 42, n_boot: int = 2000) -> pd.DataFrame:
    """Para cada produto: o prêmio mediano da Ipiranga é diferente de zero?

    Cada linha de `mercado` é um par (município, semana) em que a mediana
    Ipiranga foi comparada com a dos concorrentes ao lado. Usa Wilcoxon de
    postos sinalizados (não supõe normalidade) e um intervalo de confiança de
    95% por bootstrap para o prêmio do município típico (mediana das medianas
    de cada município).

    Cuidado de leitura: as semanas de um mesmo município não são independentes,
    então o p-valor é otimista; o intervalo por bootstrap de municípios
    (reamostrando municípios inteiros) é a medida mais honesta.
    """
    rng = np.random.default_rng(semente)
    linhas = []
    dados = mercado.dropna(subset=["premio_ipiranga_pct"])
    for produto, grupo in dados.groupby("produto"):
        premio = grupo["premio_ipiranga_pct"].to_numpy()
        if len(premio) < 30:
            continue
        teste = stats.wilcoxon(premio)
        por_municipio = grupo.groupby(["uf", "municipio"])["premio_ipiranga_pct"].median().to_numpy()
        reamostras = rng.choice(por_municipio, size=(n_boot, len(por_municipio)), replace=True)
        medianas = np.median(reamostras, axis=1)
        linhas.append(
            {
                "produto": produto,
                "comparacoes": int(len(premio)),
                "municipios": int(len(por_municipio)),
                "premio_mediano_pct": round(float(np.median(premio)), 3),
                "premio_municipio_tipico_pct": round(float(np.median(por_municipio)), 3),
                "ic95_inf_pct": round(float(np.percentile(medianas, 2.5)), 3),
                "ic95_sup_pct": round(float(np.percentile(medianas, 97.5)), 3),
                "pct_acima": round(float(np.mean(premio > 0) * 100), 1),
                "pct_igual": round(float(np.mean(premio == 0) * 100), 1),
                "pct_abaixo": round(float(np.mean(premio < 0) * 100), 1),
                # empates (prêmio 0) são descartados pelo Wilcoxon; p-valor otimista (semanas dependentes)
                "wilcoxon_p": float(teste.pvalue),
            }
        )
    return pd.DataFrame(linhas).sort_values("premio_mediano_pct", ascending=False)
