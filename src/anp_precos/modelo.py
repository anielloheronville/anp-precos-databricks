"""Modelo: a mediana de um município vai subir na semana que vem?

O Spark faz o trabalho pesado (3 milhões de coletas viram ~500 mil linhas de
município x produto x semana); o modelo roda em pandas + scikit-learn sobre a
tabela gold, que cabe em memória. Validação sempre no tempo: treina no passado,
mede no futuro, nunca embaralha semanas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NUMERICAS = [
    "var_1s",
    "var_2s",
    "var_4s",
    "var_nacional_1s",
    "var_nacional_4s",
    "dispersao",
    "n_postos",
    "premio_ipiranga_pct",
    "semana_do_ano",
]
CATEGORICAS = ["produto", "regiao"]
ALVO = "sobe_prox_semana"


@dataclass
class Particao:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame
    cortes: dict = field(default_factory=dict)


def particionar_no_tempo(df: pd.DataFrame, semanas_teste: int = 26, semanas_val: int = 26) -> Particao:
    """Últimas `semanas_teste` semanas = teste; as `semanas_val` antes = validação; resto = treino.

    Embargo de 1 semana em cada fronteira: o alvo da semana t usa o preço de
    t+1, então a última semana antes de cada corte sai, para o rótulo do treino
    não enxergar a primeira semana da validação (e o da validação, a do teste).
    """
    df = df[df[ALVO].notna()].copy()
    semanas = np.sort(df["semana"].unique())
    if len(semanas) <= semanas_teste + semanas_val + 2:
        raise ValueError("Poucas semanas para separar treino, validação e teste.")
    ini_teste = semanas[-semanas_teste]
    ini_val = semanas[-(semanas_teste + semanas_val)]
    fim_treino = semanas[-(semanas_teste + semanas_val + 1)]  # semana embargada
    fim_val = semanas[-(semanas_teste + 1)]  # semana embargada
    return Particao(
        treino=df[df["semana"] < fim_treino],
        validacao=df[(df["semana"] >= ini_val) & (df["semana"] < fim_val)],
        teste=df[df["semana"] >= ini_teste],
        cortes={"inicio_validacao": str(ini_val)[:10], "inicio_teste": str(ini_teste)[:10]},
    )


def _matriz(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NUMERICAS].astype("float64").copy()
    for coluna in CATEGORICAS:
        x[coluna] = df[coluna].astype("category")
    return x


def _categorias_fixas(*partes: pd.DataFrame) -> dict:
    return {c: sorted(pd.concat([p[c] for p in partes]).dropna().unique()) for c in CATEGORICAS}


def _aplicar_categorias(x: pd.DataFrame, categorias: dict) -> pd.DataFrame:
    for coluna, valores in categorias.items():
        x[coluna] = pd.Categorical(x[coluna], categories=valores)
    return x


def _numericas_logistica(x: pd.DataFrame) -> pd.DataFrame:
    """A logística não aceita nulo: zera, mas marca onde não havia comparação de prêmio."""
    saida = x[NUMERICAS].copy()
    saida["tem_premio"] = saida["premio_ipiranga_pct"].notna().astype("float64")
    return saida.fillna(0.0)


def _precisao_no_topo(y: np.ndarray, score: np.ndarray, fracao: float = 0.1) -> float:
    """Dos 10% de mercados com maior score, quantos de fato subiram."""
    k = max(1, int(len(score) * fracao))
    topo = np.argsort(-score)[:k]
    return float(np.mean(y[topo]))


def _metricas(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "average_precision": round(float(average_precision_score(y, score)), 4),
        "precisao_top10pct": round(_precisao_no_topo(y, score), 4),
    }


def treinar_e_avaliar(df: pd.DataFrame, semente: int = 42) -> dict:
    """Treina o gradient boosting, compara com duas referências e devolve tudo.

    A validação (semanas logo antes do teste) decide quando o boosting para
    (early stopping), sem amostra aleatória de linhas: embaralhar misturaria
    linhas da mesma semana, que dividem o mesmo choque nacional de preço.

    Referências: (1) persistência, a variação da última semana como score;
    (2) regressão logística nas mesmas features numéricas. O TESTE não é usado
    para nenhuma escolha.
    """
    p = particionar_no_tempo(df)
    categorias = _categorias_fixas(p.treino, p.validacao, p.teste)
    xs = {nome: _aplicar_categorias(_matriz(parte), categorias) for nome, parte in
          [("treino", p.treino), ("validacao", p.validacao), ("teste", p.teste)]}
    ys = {nome: parte[ALVO].astype(int).to_numpy() for nome, parte in
          [("treino", p.treino), ("validacao", p.validacao), ("teste", p.teste)]}

    modelo = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        n_iter_no_change=30,
        scoring="roc_auc",
        random_state=semente,
    )
    modelo.fit(xs["treino"], ys["treino"], X_val=xs["validacao"], y_val=ys["validacao"])

    logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logistica.fit(_numericas_logistica(xs["treino"]), ys["treino"])

    resultado = {"cortes": p.cortes, "linhas": {k: int(len(v)) for k, v in ys.items()}}
    resultado["taxa_de_alta"] = {k: round(float(v.mean()), 4) for k, v in ys.items()}
    for nome in ["validacao", "teste"]:
        x, y = xs[nome], ys[nome]
        persistencia = x["var_1s"].fillna(0.0).to_numpy()
        resultado[nome] = {
            "gradient_boosting": _metricas(y, modelo.predict_proba(x)[:, 1]),
            "logistica": _metricas(y, logistica.predict_proba(_numericas_logistica(x))[:, 1]),
            "persistencia": _metricas(y, persistencia),
        }
    resultado["_modelo"] = modelo
    resultado["_x_teste"] = xs["teste"]
    resultado["_y_teste"] = ys["teste"]
    return resultado


def importancia_por_permutacao(resultado: dict, n_repeticoes: int = 5, semente: int = 42) -> dict:
    """Quanto o AUC de teste cai ao embaralhar cada feature."""
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(
        resultado["_modelo"],
        resultado["_x_teste"],
        resultado["_y_teste"],
        scoring="roc_auc",
        n_repeats=n_repeticoes,
        random_state=semente,
    )
    nomes = list(resultado["_x_teste"].columns)
    pares = sorted(zip(nomes, imp.importances_mean), key=lambda kv: -kv[1])
    return {nome: round(float(valor), 4) for nome, valor in pares}
