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
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(nomes))))
    ax.barh(nomes, valores, color="#d95f02")
    ax.set_xlabel("Queda no ROC AUC ao embaralhar a feature")
    ax.set_title("Importância por permutação (teste)")
    ax.grid(axis="x", alpha=0.3)
    return _salvar(fig, pasta, "importancia_features.png")


# ---------------------------------------------------------------------------------------------
# Barril, bomba, cenários e eventos
# ---------------------------------------------------------------------------------------------
COR_BOMBA, COR_BARRIL, COR_DIESEL, COR_NEUTRA = "#d95f02", "#7570b3", "#1b9e77", "#555555"
RODAPE = {"fontsize": 7.5, "color": COR_NEUTRA}


def barril_e_bomba(semanal: pd.DataFrame, pasta: Path, corte: str, inicio: str = "2017-01-01") -> Path:
    """Níveis em R$/litro e variação em 4 semanas do barril e da gasolina, na mesma escala."""
    import matplotlib.dates as mdates

    dados = semanal.copy()
    dados["semana"] = pd.to_datetime(dados["semana"])
    dados = dados.sort_values("semana").set_index("semana")
    dados = dados.reindex(pd.date_range(dados.index.min(), dados.index.max(), freq="7D"))
    dados = dados[dados.index >= inicio]
    fig, eixos = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax = eixos[0]
    ax.plot(dados.index, dados["gasolina"], color=COR_BOMBA, linewidth=2, label="Gasolina comum na bomba (Brasil)")
    ax.plot(dados.index, dados["diesel_s10"], color=COR_DIESEL, linewidth=1.1, label="Diesel S10 na bomba (Brasil)")
    ax.plot(
        dados.index,
        dados["brent_reais_litro"],
        color=COR_BARRIL,
        linewidth=1.6,
        label="Barril de Brent em reais, por litro (Brent × dólar ÷ 159)",
    )
    ax.set_ylabel("R$ por litro")
    ax.set_ylim(0, None)
    ax.set_title("Barril de Brent em reais e preço na bomba", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    sem_pesquisa = (dados["gasolina"].isna() & dados["brent_reais_litro"].notna()).to_numpy()
    for a in eixos:
        a.axvline(pd.Timestamp(corte), color=COR_NEUTRA, linestyle=":", linewidth=1)
        a.grid(alpha=0.3)
        a.fill_between(
            dados.index, 0, 1, where=sem_pesquisa, transform=a.get_xaxis_transform(), color="#bbbbbb", alpha=0.35,
            linewidth=0,
        )
    ax.text(pd.Timestamp(corte), 0.25, " mai/2023: Petrobras\n deixa a paridade de importação", va="bottom", **RODAPE)

    variacoes = {
        "Barril em reais": (100 * (dados["brent_reais_litro"] / dados["brent_reais_litro"].shift(4) - 1), COR_BARRIL),
        "Gasolina na bomba": (100 * (dados["gasolina"] / dados["gasolina"].shift(4) - 1), COR_BOMBA),
    }
    limite = 70  # mesma escala nos dois painéis, de propósito; o que passa disso é anotado
    for a, (rotulo, (serie, cor)) in zip(eixos[1:], variacoes.items()):
        a.fill_between(serie.index, 0, serie, color=cor, alpha=0.55, linewidth=0)
        a.plot(serie.index, serie, color=cor, linewidth=0.8)
        a.axhline(0, color="black", linewidth=0.6)
        a.set_ylabel("variação em\n4 semanas (%)")
        a.set_title(rotulo, loc="left", fontsize=10)
        a.set_ylim(-limite, limite)
        if serie.max() > limite:
            a.annotate(
                f"+{serie.max():.0f}% em {serie.idxmax():%m/%Y}\n(fora da escala)",
                xy=(serie.idxmax(), limite),
                xytext=(serie.idxmax() + pd.Timedelta(weeks=20), limite * 0.62),
                arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.7},
                **RODAPE,
            )
    eixos[2].xaxis.set_major_locator(mdates.YearLocator())
    eixos[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(
        0.01,
        0.005,
        "Fontes: ANP (levantamento semanal de preços, Brasil), U.S. EIA pelo FRED (Brent) e Banco Central (PTAX). "
        "Faixa cinza: semanas sem pesquisa da ANP.",
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _salvar(fig, pasta, "barril_e_bomba.png")


def curvas_de_repasse(curvas: dict[str, dict[str, dict]], pasta: Path) -> Path:
    """Repasse acumulado por produto (um painel cada), comparando os períodos.

    `curvas`: {rótulo do produto: {rótulo do período: resultado de repasse.curva}}.
    """
    fig, eixos = plt.subplots(1, len(curvas), figsize=(5.8 * len(curvas), 4.6), sharey=True, squeeze=False)
    estilos = [(COR_NEUTRA, "--"), (COR_BOMBA, "-"), (COR_DIESEL, "-.")]
    for ax, (produto, periodos) in zip(eixos[0], curvas.items()):
        for (periodo, curva), (cor, traco) in zip(periodos.items(), estilos):
            semanas = np.arange(len(curva["acumulado"]))
            rotulo = f"{periodo} ({curva['semanas']} semanas)"
            ax.plot(semanas, curva["acumulado"], color=cor, linestyle=traco, linewidth=2, label=rotulo)
            ax.fill_between(semanas, curva["ic90_inf"], curva["ic90_sup"], color=cor, alpha=0.15, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(produto, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("semanas depois da alta do barril")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    eixos[0][0].set_ylabel("alta acumulada na bomba (%)\npara cada 1% de alta do barril em reais")
    fig.suptitle(
        "Quanto do barril chega à bomba, e em quantas semanas", x=0.01, ha="left", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.01,
        0.005,
        "Regressão com defasagens semanais das variações em log; faixa = intervalo de 90% por bootstrap em blocos de "
        "8 semanas. Não separa mudanças de imposto.",
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _salvar(fig, pasta, "repasse_do_barril.png")


def cenarios(tabela: pd.DataFrame, pasta: Path, semanas_depois: int = 16) -> Path:
    """Preço na bomba em cada cenário, com o intervalo de 90%, um painel por curva de repasse."""
    dados = tabela[tabela["semanas_depois"] == semanas_depois]
    curvas = list(dict.fromkeys(dados["curva"]))
    nomes = list(dict.fromkeys(dados["cenario"]))
    fig, eixos = plt.subplots(
        1, len(curvas), figsize=(4.6 * len(curvas), 1.2 + 0.8 * len(nomes)), sharey=True, squeeze=False
    )
    posicoes = np.arange(len(nomes))[::-1]
    for ax, curva in zip(eixos[0], curvas):
        parte = dados[dados["curva"] == curva].set_index("cenario").loc[nomes]
        cor = COR_DIESEL if "diesel" in parte["produto"].iloc[0] else COR_BOMBA
        largura = parte["preco_ic90_sup"] - parte["preco_ic90_inf"]
        ax.barh(posicoes, largura, left=parte["preco_ic90_inf"], color=cor, alpha=0.25, height=0.5, linewidth=0)
        ax.scatter(parte["preco"], posicoes, color=cor, zorder=3, s=36)
        ax.axvline(parte["preco_atual"].iloc[0], color="black", linewidth=0.8, linestyle=":")
        for y, (_, linha) in zip(posicoes, parte.iterrows()):
            # "\$": sem a barra, o Matplotlib lê o texto entre dois cifrões como fórmula
            texto = f"  R\\$ {linha['preco']:.2f} ({linha['variacao_pct']:+.1f}%)"
            ax.text(max(linha["preco_ic90_sup"], linha["preco"]), y, texto, va="center", fontsize=8)
        ax.set_title(curva, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("R\\$ por litro")
        ax.grid(axis="x", alpha=0.3)
        minimo = min(parte["preco_ic90_inf"].min(), parte["preco"].min())
        maximo = max(parte["preco_ic90_sup"].max(), parte["preco"].max())
        folga = max(maximo - minimo, 0.05)
        ax.set_xlim(minimo - 0.08 * folga, maximo + 0.8 * folga)  # espaço à direita para o rótulo de cada barra
    referencia = dados.drop_duplicates("cenario").set_index("cenario")
    eixos[0][0].set_yticks(posicoes)
    rotulos = [
        f"{nome}\n(Brent US\\$ {referencia.at[nome, 'brent_usd']:.0f}, dólar R\\$ {referencia.at[nome, 'dolar']:.2f})"
        for nome in nomes
    ]
    eixos[0][0].set_yticklabels(rotulos, fontsize=8.5)
    fig.suptitle(
        f"Preço na bomba {semanas_depois} semanas depois do salto do barril, em cada cenário", x=0.01, ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Ponto = estimativa; barra = intervalo de 90%; linha pontilhada = preço de hoje. Só o efeito do barril: sem "
        "imposto, sem tendência do preço e sem mudança na política de preços.",
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return _salvar(fig, pasta, "cenarios_bomba.png")


def eventos_e_alimentos(mensal: pd.DataFrame, eventos: list[tuple], pasta: Path, inicio: str = "2018-01") -> Path:
    """Variação em 12 meses de alimentos, dólar, barril e diesel, com os eventos marcados.

    `mensal` é a tabela já preparada por eventos.preparar (índice mensal).
    """
    import matplotlib.dates as mdates

    tempo = mensal.index.to_timestamp()

    def em_12_meses(coluna: str) -> pd.Series:
        return 100 * (mensal[coluna] / mensal[coluna].shift(12) - 1)

    fig, eixos = plt.subplots(3, 1, figsize=(11, 9.8), sharex=True)
    eixos[0].plot(
        tempo, em_12_meses("nivel_ipca_alimentos"), color=COR_BOMBA, linewidth=2.2,
        label="Alimentos e bebidas no Brasil (IPCA)",
    )
    eixos[0].plot(
        tempo, em_12_meses("nivel_ipca_geral"), color=COR_NEUTRA, linewidth=1.3, linestyle="--",
        label="Inflação geral (IPCA)",
    )
    eixos[0].set_title(
        "Alimentos, dólar, barril e diesel em 12 meses, com os eventos marcados",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    eixos[1].plot(
        tempo, em_12_meses("alimentos_mundo_reais"), color=COR_DIESEL, linewidth=2,
        label="Alimentos no mundo, em reais (índice FMI × dólar)",
    )
    eixos[1].plot(
        tempo, em_12_meses("alimentos_mundo"), color=COR_DIESEL, linewidth=1.1, linestyle=":",
        label="Alimentos no mundo, em dólares",
    )
    eixos[1].plot(tempo, em_12_meses("dolar"), color=COR_BARRIL, linewidth=1.3, label="Dólar")
    eixos[2].plot(
        tempo, em_12_meses("barril_reais").clip(upper=130), color=COR_BARRIL, linewidth=1.3,
        label="Barril de Brent em reais (cortado em +130%)",
    )
    eixos[2].plot(tempo, em_12_meses("diesel_s10"), color=COR_BOMBA, linewidth=2, label="Diesel S10 na bomba")
    ancoras = [(0.0, 1.0), (0.60, 1.0), (0.50, 1.0)]
    for i, ax in enumerate(eixos):
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("variação em 12 meses (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=ancoras[i])
        for nome, base, _ in eventos:
            quando = (pd.Period(base, freq="M") + 1).to_timestamp()
            ax.axvline(quando, color="#444444", linestyle=":", linewidth=1)
            if i == 0:
                ax.annotate(
                    nome, xy=(quando, 0.02), xycoords=("data", "axes fraction"), xytext=(3, 0),
                    textcoords="offset points", fontsize=8.5, color="#333333", va="bottom", rotation=90,
                )
    eixos[2].xaxis.set_major_locator(mdates.YearLocator())
    eixos[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    eixos[2].set_xlim(pd.Timestamp(inicio), tempo.max() + pd.Timedelta(days=45))
    fig.text(
        0.01,
        0.005,
        "Fontes: IBGE pelo SGS do Banco Central (IPCA), FMI pelo FRED (alimentos no mundo), Banco Central (PTAX), "
        "U.S. EIA pelo FRED (Brent), ANP (diesel).",
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _salvar(fig, pasta, "eventos_e_alimentos.png")
