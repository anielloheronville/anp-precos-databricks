"""Gráficos em Matplotlib para o relatório (funcionam local e no Databricks)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Paleta neutra (não são as cores das marcas); a bandeira em foco ganha destaque.
CORES = {
    "IPIRANGA": "#d95f02",
    "VIBRA": "#1b9e77",
    "RAIZEN": "#7570b3",
    "BRANCA": "#7f7f7f",
    "ALE": "#e7298a",
    "OUTRAS": "#bdbdbd",
}


def _salvar(fig, pasta: Path, nome: str) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    fig.savefig(caminho, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return caminho


def evolucao_por_bandeira(bandeira_semanal: pd.DataFrame, produto: str, pasta: Path) -> Path:
    dados = bandeira_semanal[bandeira_semanal["produto"] == produto]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for bandeira, grupo in dados.groupby("bandeira"):
        grupo = grupo.sort_values("semana")
        ax.plot(
            pd.to_datetime(grupo["semana"]),
            grupo["preco_mediano"],
            label=bandeira,
            color=CORES.get(bandeira, "#444"),
            linewidth=2.2 if bandeira == "IPIRANGA" else 1.2,
        )
    ax.set_title(f"{produto}: preço mediano nacional por bandeira (R$/L)")
    ax.set_ylabel("R$")
    ax.grid(alpha=0.3)
    ax.legend(ncol=6, fontsize=8, loc="upper left")
    return _salvar(fig, pasta, f"evolucao_{produto.lower().replace(' ', '_')}.png")


def premio_por_uf(mercado: pd.DataFrame, produto: str, pasta: Path, min_comparacoes: int = 20) -> Path:
    dados = mercado[(mercado["produto"] == produto) & mercado["premio_ipiranga_pct"].notna()]
    # média, não mediana: muitos mercados empatam no centavo e a mediana vira 0 em metade das UFs
    por_uf = dados.groupby("uf")["premio_ipiranga_pct"].agg(["mean", "count"])
    por_uf = por_uf[por_uf["count"] >= min_comparacoes].sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(por_uf))))
    cores = ["#d95f02" if v > 0 else "#1b9e77" for v in por_uf["mean"]]
    ax.barh(por_uf.index, por_uf["mean"], color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Prêmio médio da Ipiranga vs. concorrentes do mesmo município (%)")
    ax.set_title(f"{produto}: onde a Ipiranga está acima ou abaixo do mercado")
    ax.grid(axis="x", alpha=0.3)
    return _salvar(fig, pasta, f"premio_uf_{produto.lower().replace(' ', '_')}.png")


def qualidade(resumo: dict, pasta: Path) -> Path:
    motivos = resumo["motivos"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if motivos:
        nomes = list(motivos)[::-1]
        ax.barh(nomes, [motivos[n] for n in nomes], color="#7570b3")
        for i, nome in enumerate(nomes):
            ax.text(motivos[nome], i, f" {motivos[nome]:,}".replace(",", "."), va="center", fontsize=8)
    ax.set_title(
        f"Quarentena: {resumo['linhas_quarentena']:,} de {resumo['linhas_bronze']:,} linhas".replace(",", ".")
    )
    ax.set_xlabel("linhas")
    return _salvar(fig, pasta, "qualidade_quarentena.png")


def curva_roc(resultado: dict, pasta: Path) -> Path:
    y = resultado["_y_teste"]
    x = resultado["_x_teste"]
    fig, ax = plt.subplots(figsize=(5, 5))
    scores = {
        "Gradient boosting": resultado["_modelo"].predict_proba(x)[:, 1],
        "Persistência (var. última semana)": x["var_1s"].fillna(0.0).to_numpy(),
    }
    for nome, score in scores.items():
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=nome)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999", linewidth=0.8)
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdadeiros positivos")
    ax.set_title("Alta na próxima semana: teste (últimas 26 semanas)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    return _salvar(fig, pasta, "roc_teste.png")


def importancia(importancias: dict, pasta: Path) -> Path:
    nomes = list(importancias)[::-1]
    valores = np.array([importancias[n] for n in nomes])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(nomes, valores, color="#d95f02")
    ax.set_xlabel("Queda no ROC AUC ao embaralhar a feature")
    ax.set_title("Importância por permutação (teste)")
    ax.grid(axis="x", alpha=0.3)
    return _salvar(fig, pasta, "importancia_features.png")
