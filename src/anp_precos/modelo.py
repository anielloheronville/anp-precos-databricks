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

from anp_precos.esquema import COLUNAS_BRENT

NUMERICAS_BASE = [
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
# Variação do barril de Brent em reais em 1, 4 e 8 semanas (iguais para todos os municípios da semana).
NUMERICAS_BRENT = list(COLUNAS_BRENT)
NUMERICAS = NUMERICAS_BASE + NUMERICAS_BRENT
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


def _matriz(df: pd.DataFrame, numericas: list[str]) -> pd.DataFrame:
    x = df[numericas].astype("float64").copy()
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
    saida = x.drop(columns=CATEGORICAS).copy()
    saida["tem_premio"] = saida["premio_ipiranga_pct"].notna().astype("float64")
    return saida.fillna(0.0)


def _por_produto(produtos: pd.Series, y: np.ndarray, score: np.ndarray) -> dict:
    """ROC AUC e taxa de alta de cada produto (o repasse do barril não é igual entre eles)."""
    saida = {}
    for produto in sorted(produtos.dropna().unique()):
        mascara = (produtos == produto).to_numpy()
        if len(np.unique(y[mascara])) < 2:
            continue
        saida[str(produto)] = {
            "linhas": int(mascara.sum()),
            "taxa_de_alta": round(float(y[mascara].mean()), 4),
            "roc_auc": round(float(roc_auc_score(y[mascara], score[mascara])), 4),
        }
    return saida


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


def treinar_e_avaliar(df: pd.DataFrame, semente: int = 42, numericas: list[str] | None = None) -> dict:
    """Treina o gradient boosting, compara com duas referências e devolve tudo.

    A validação (semanas logo antes do teste) decide quando o boosting para
    (early stopping), sem amostra aleatória de linhas: embaralhar misturaria
    linhas da mesma semana, que dividem o mesmo choque nacional de preço.

    Referências: (1) persistência, a variação da última semana como score;
    (2) regressão logística nas mesmas features numéricas. O TESTE não é usado
    para nenhuma escolha.

    `numericas` troca a lista de features numéricas (padrão: todas, com o Brent).
    """
    numericas = list(NUMERICAS if numericas is None else numericas)
    faltando = sorted(set(numericas + CATEGORICAS + [ALVO, "semana"]) - set(df.columns))
    if faltando:
        raise ValueError(
            f"A tabela de features não tem as colunas {faltando}. "
            "Se for o gold_features de uma versão anterior, rode o pipeline (notebook 01) de novo."
        )
    p = particionar_no_tempo(df)
    categorias = _categorias_fixas(p.treino, p.validacao, p.teste)
    xs = {nome: _aplicar_categorias(_matriz(parte, numericas), categorias) for nome, parte in
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
    scores = {}
    for nome in ["validacao", "teste"]:
        x, y = xs[nome], ys[nome]
        persistencia = x["var_1s"].fillna(0.0).to_numpy()
        scores[nome] = modelo.predict_proba(x)[:, 1]
        resultado[nome] = {
            "gradient_boosting": _metricas(y, scores[nome]),
            "logistica": _metricas(y, logistica.predict_proba(_numericas_logistica(x))[:, 1]),
            "persistencia": _metricas(y, persistencia),
        }
    resultado["teste_por_produto"] = _por_produto(p.teste["produto"], ys["teste"], scores["teste"])
    resultado["_modelo"] = modelo
    resultado["_x_teste"] = xs["teste"]
    resultado["_y_teste"] = ys["teste"]
    # para comparar dois modelos linha a linha (mesma partição, mesma ordem)
    resultado["_scores"] = scores
    resultado["_ys"] = {nome: ys[nome] for nome in scores}
    resultado["_semanas"] = {"validacao": p.validacao["semana"].to_numpy(), "teste": p.teste["semana"].to_numpy()}
    return resultado


def _ganho_auc_por_semana(
    semanas: np.ndarray, y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n: int = 400, semente: int = 42
) -> dict:
    """AUC(a) - AUC(b) com intervalo de 90% reamostrando SEMANAS inteiras.

    As features do barril são iguais para todos os municípios da semana, então
    a unidade independente é a semana, não a linha: reamostrar linhas daria um
    intervalo estreito demais.
    """
    rng = np.random.default_rng(semente)
    _, codigo = np.unique(semanas, return_inverse=True)
    por_semana = [np.flatnonzero(codigo == i) for i in range(codigo.max() + 1)]
    diferencas = []
    for _ in range(n):
        idx = np.concatenate([por_semana[i] for i in rng.integers(0, len(por_semana), len(por_semana))])
        if len(np.unique(y[idx])) < 2:
            continue
        diferencas.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
    inferior, superior = np.percentile(diferencas, [5, 95])
    return {
        "diferenca": round(float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)), 4),
        "ic90": [round(float(inferior), 4), round(float(superior), 4)],
        "semanas": int(len(por_semana)),
    }


def comparar_sem_brent(df: pd.DataFrame, resultado: dict, semente: int = 42) -> dict:
    """O barril em reais ajuda? Treina de novo sem as features do Brent e põe lado a lado.

    Mesma partição, mesma semente e mesmos hiperparâmetros: a única diferença
    entre as duas colunas são as três variações do barril. `ganho_auc` traz a
    diferença do gradient boosting com intervalo por bootstrap de semanas. A
    decisão de manter a feature deve olhar a VALIDAÇÃO; o teste só confirma.
    """
    sem = treinar_e_avaliar(df, semente=semente, numericas=NUMERICAS_BASE)
    comparacao: dict = {}
    for particao in ["validacao", "teste"]:
        comparacao[particao] = {
            nome: {"com_brent": resultado[particao][nome], "sem_brent": sem[particao][nome]}
            for nome in ["gradient_boosting", "logistica"]
        }
    comparacao["ganho_auc_gradient_boosting"] = {
        particao: _ganho_auc_por_semana(
            resultado["_semanas"][particao],
            resultado["_ys"][particao],
            resultado["_scores"][particao],
            sem["_scores"][particao],
            semente=semente,
        )
        for particao in ["validacao", "teste"]
    }
    comparacao["teste_por_produto"] = {
        produto: {
            "linhas": valores["linhas"],
            "taxa_de_alta": valores["taxa_de_alta"],
            "roc_auc_com_brent": valores["roc_auc"],
            "roc_auc_sem_brent": sem["teste_por_produto"].get(produto, {}).get("roc_auc"),
        }
        for produto, valores in resultado["teste_por_produto"].items()
    }
    return comparacao


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
