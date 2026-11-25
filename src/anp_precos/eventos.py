"""Estudo de eventos: o que aconteceu com dólar, barril, bomba e alimentos depois de cada choque.

Entrada: a tabela mensal (`gold_macro_mensal`). Tudo aqui é descritivo. Um evento
é uma data, não uma variável: a tabela mostra o que mudou depois dele, e as
correlações mostram por quais canais os preços costumam andar juntos. Nada disso
prova causa, e são cerca de 115 meses, com vários choques sobrepostos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anp_precos.esquema import EVENTOS

# rótulo de leitura -> coluna de nível (preço ou índice) na tabela mensal preparada
NIVEIS = {
    "Dólar": "dolar",
    "Barril em reais": "barril_reais",
    "Diesel S10 na bomba": "diesel_s10",
    "Gasolina na bomba": "gasolina",
    "Alimentos no mundo (US$)": "alimentos_mundo",
    "Alimentos no mundo (R$)": "alimentos_mundo_reais",
    "Alimentos no Brasil (IPCA)": "nivel_ipca_alimentos",
    "Inflação geral (IPCA)": "nivel_ipca_geral",
}
CANAIS = ["Alimentos no mundo (R$)", "Diesel S10 na bomba"]
PERIODOS = {
    "antes da COVID (até fev/2020)": (None, "2020-02"),
    "COVID e Ucrânia (mar/2020 a dez/2022)": ("2020-03", "2022-12"),
    "de 2023 em diante": ("2023-01", None),
}


def preparar(mensal: pd.DataFrame, inicio: str = "2017-01") -> pd.DataFrame:
    """Índice mensal regular e níveis de preço a partir das variações do IPCA.

    O IPCA chega como variação mensal em %; acumulado, vira um índice
    comparável com as outras séries, que são níveis. O índice começa no primeiro
    mês em que a série existe (base 100 no mês anterior) e para de acumular no
    primeiro buraco: não emenda por cima de mês faltante.

    A tabela termina no último mês com IPCA divulgado. O mês corrente, ainda
    aberto, teria dólar e barril pela média parcial e ficaria sem IPCA.
    """
    dados = mensal.copy()
    dados["mes"] = pd.to_datetime(dados["mes"]).dt.to_period("M")
    dados = dados.dropna(subset=["mes"]).sort_values("mes").drop_duplicates("mes").set_index("mes")
    dados = dados[dados.index >= pd.Period(inicio, freq="M")]
    fechado = dados["ipca_geral"].last_valid_index()
    if fechado is None:
        raise ValueError("A tabela mensal não tem nenhum mês com IPCA: confira os arquivos em referencias/.")
    dados = dados.reindex(pd.period_range(dados.index.min(), fechado, freq="M", name="mes"))
    for coluna in ["ipca_alimentos", "ipca_geral"]:
        fator = 1 + dados[coluna] / 100
        primeiro = fator.first_valid_index()
        nivel = pd.Series(np.nan, index=dados.index)
        if primeiro is not None:
            trecho = fator.loc[primeiro:]
            nivel.loc[primeiro:] = (100 * trecho.cumprod()).where(trecho.notna().cummin())
        dados[f"nivel_{coluna}"] = nivel
    return dados


def _variacao(dados: pd.DataFrame, coluna: str, de: pd.Period, ate: pd.Period) -> float | None:
    if de not in dados.index or ate not in dados.index:
        return None
    antes, depois = dados.at[de, coluna], dados.at[ate, coluna]
    return None if pd.isna(antes) or pd.isna(depois) else round(float(100 * (depois / antes - 1)), 1)


def tabela(mensal: pd.DataFrame, eventos: list[tuple] = EVENTOS, janelas: tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
    """Variação de cada série nos 12 meses antes do mês-base e nos meses seguintes a cada evento."""
    dados = preparar(mensal)
    linhas = []
    for nome, base, descricao in eventos:
        mes_base = pd.Period(base, freq="M")
        for rotulo, coluna in NIVEIS.items():
            linha = {"evento": nome, "mes_base": base, "descricao": descricao, "serie": rotulo}
            linha["12 meses antes"] = _variacao(dados, coluna, mes_base - 12, mes_base)
            for janela in janelas:
                linha[f"+{janela} meses"] = _variacao(dados, coluna, mes_base, mes_base + janela)
            linhas.append(linha)
    return pd.DataFrame(linhas)


def _variacoes_mensais(dados: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {rotulo: 100 * np.log(dados[coluna] / dados[coluna].shift(1))
         for rotulo, coluna in NIVEIS.items() if not coluna.startswith("nivel_")}
    )


def correlacoes(mensal: pd.DataFrame, defasagens: int = 6) -> pd.DataFrame:
    """Correlação do IPCA de alimentos do mês t com a variação de cada série k meses antes."""
    dados = preparar(mensal)
    variacoes = _variacoes_mensais(dados)
    alvo = dados["ipca_alimentos"]
    return pd.DataFrame(
        {k: {rotulo: round(float(alvo.corr(variacoes[rotulo].shift(k))), 2) for rotulo in variacoes} for k in
         range(defasagens + 1)}
    ).rename_axis("serie").rename_axis("meses_antes", axis=1)


def canais(mensal: pd.DataFrame, defasagens: int = 4) -> dict:
    """Quanto da inflação de alimentos os canais medidos explicam, no todo e em cada período.

    Regressão do IPCA de alimentos nas variações, do mês e dos `defasagens`
    anteriores, dos alimentos no mundo em reais e do diesel na bomba.

    A ANP não pesquisou preços em setembro de 2020. Sem tratamento, esse único
    mês sem diesel tiraria da conta setembro de 2020 a fevereiro de 2021,
    justamente os meses de maior inflação de alimentos da série. Um buraco de
    um mês no canal é interpolado (em log) entre os vizinhos; buraco maior não.

    Por período: a inflação de alimentos de TODOS os meses do período, e a
    contribuição média dos dois canais (sem o intercepto) nos meses que entram
    na regressão. O intercepto é a inflação de alimentos que não depende deles.
    """
    dados = preparar(mensal)
    niveis = dados[[NIVEIS[canal] for canal in CANAIS]]
    interpolado = np.exp(np.log(niveis).interpolate(limit=1, limit_area="inside"))
    interpolados = int((niveis.isna() & interpolado.notna()).any(axis=1).sum())
    variacoes = pd.DataFrame(
        {canal: 100 * np.log(interpolado[NIVEIS[canal]] / interpolado[NIVEIS[canal]].shift(1)) for canal in CANAIS}
    )
    alvo = dados["ipca_alimentos"]
    x = pd.concat({(canal, k): variacoes[canal].shift(k) for canal in CANAIS for k in range(defasagens + 1)}, axis=1)
    completos = alvo.notna() & x.notna().all(axis=1)
    observados, regressores = int(completos.sum()), x.shape[1]
    matriz = np.column_stack([np.ones(observados), x[completos].to_numpy()])
    coeficientes, *_ = np.linalg.lstsq(matriz, alvo[completos].to_numpy(), rcond=None)
    contribuicao = pd.Series(matriz[:, 1:] @ coeficientes[1:], index=alvo[completos].index)
    residuo = alvo[completos] - coeficientes[0] - contribuicao
    r2 = 1 - float((residuo**2).sum()) / float(((alvo[completos] - alvo[completos].mean()) ** 2).sum())

    por_periodo = {}
    for nome, (de, ate) in PERIODOS.items():
        dentro = pd.Series(True, index=dados.index)
        if de:
            dentro &= dados.index >= pd.Period(de, freq="M")
        if ate:
            dentro &= dados.index <= pd.Period(ate, freq="M")
        todos, na_regressao = dentro & alvo.notna(), dentro & completos
        if not todos.any():
            continue
        media = float(alvo[todos].mean())
        por_periodo[nome] = {
            "meses": int(todos.sum()),
            "alimentos_pct_ao_mes": round(media, 2),
            "alimentos_pct_ao_ano": round(((1 + media / 100) ** 12 - 1) * 100, 1),
            "meses_na_regressao": int(na_regressao.sum()),
            "contribuicao_dos_canais_pct_ao_mes": (
                round(float(contribuicao[na_regressao[completos]].mean()), 2) if na_regressao.any() else None
            ),
        }
    passo = defasagens + 1
    return {
        "meses": observados,
        "meses_com_canal_interpolado": interpolados,
        "defasagens": defasagens,
        "r2": round(r2, 3),
        "r2_ajustado": round(1 - (1 - r2) * (observados - 1) / (observados - regressores - 1), 3),
        "intercepto_pct_ao_mes": round(float(coeficientes[0]), 2),
        "soma_dos_coeficientes": {
            canal: round(float(coeficientes[1 + i * passo : 1 + (i + 1) * passo].sum()), 3)
            for i, canal in enumerate(CANAIS)
        },
        "por_periodo": por_periodo,
    }
