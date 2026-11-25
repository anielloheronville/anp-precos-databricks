"""Quanto do barril chega à bomba, em quantas semanas, e o que isso implica em cenários.

Entrada: a tabela semanal do repasse (`gold_repasse_semanal`), com o preço médio na
bomba por produto e o barril de Brent em reais por litro. Tudo em pandas e NumPy:
são algumas centenas de semanas.

O modelo é uma regressão com defasagens distribuídas sobre variações semanais:

    var_bomba(t) = a + b0 * var_barril(t) + b1 * var_barril(t-1) + ... + bK * var_barril(t-K) + erro

A soma b0 + ... + bk é o repasse acumulado k semanas depois de 1% de alta do barril.
Não separa mudanças de imposto nem decisões de preço da Petrobras: mede o que
aconteceu em média no período, não uma lei.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anp_precos.esquema import CORTE_PARIDADE, DEFASAGENS_REPASSE, LITROS_POR_BARRIL, MIN_DIAS_SEMANA

BARRIL = "brent_reais_litro"
ROTULOS = {"gasolina": "Gasolina comum", "diesel_s10": "Diesel S10", "diesel": "Diesel", "etanol": "Etanol"}
ATE_O_CORTE, DEPOIS_DO_CORTE = "até mai/2023 (paridade de importação)", "de mai/2023 em diante"
PERIODOS = {ATE_O_CORTE: (None, CORTE_PARIDADE), DEPOIS_DO_CORTE: (CORTE_PARIDADE, None)}
# Se a bomba estiver mais do que isto atrás do barril, os cenários não são calculados.
MAX_ATRASO_DA_BOMBA = 2  # semanas


def grade_semanal(semanal: pd.DataFrame) -> pd.DataFrame:
    """Índice semanal regular, para que "k semanas antes" seja sempre k linhas antes.

    Semana sem pesquisa da ANP (2020) ou sem barril vira linha com nulos, e toda
    variação que tocaria nela fica nula, em vez de pular para a semana vizinha.
    """
    dados = semanal.copy()
    dados["semana"] = pd.to_datetime(dados["semana"])
    dados = dados.dropna(subset=["semana"]).sort_values("semana").drop_duplicates("semana").set_index("semana")
    return dados.reindex(pd.date_range(dados.index.min(), dados.index.max(), freq="7D", name="semana"))


def variacao_log(serie: pd.Series, passos: int = 1) -> pd.Series:
    """Variação em % (logarítmica) contra `passos` linhas antes."""
    return 100 * np.log(serie / serie.shift(passos))


def _matriz(grade: pd.DataFrame, produto: str, defasagens: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    y = variacao_log(grade[produto])
    barril = variacao_log(grade[BARRIL])
    x = pd.concat({k: barril.shift(k) for k in range(defasagens + 1)}, axis=1)
    completas = y.notna() & x.notna().all(axis=1)
    return x[completas].to_numpy(), y[completas].to_numpy(), y[completas].index


def _betas(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeficientes, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), x]), y, rcond=None)
    return coeficientes[1:]  # sem o intercepto


def curva(
    semanal: pd.DataFrame,
    produto: str,
    inicio: str | None = None,
    fim: str | None = None,
    defasagens: int = DEFASAGENS_REPASSE,
    n_boot: int = 1000,
    bloco: int = 8,
    semente: int = 42,
) -> dict:
    """Repasse acumulado do barril para `produto`, de 0 a `defasagens` semanas.

    `inicio` e `fim` (fim exclusivo) recortam as semanas da bomba; as defasagens
    do barril podem vir de antes do recorte. O intervalo de 90% vem de bootstrap
    em blocos de `bloco` semanas, porque variações vizinhas não são independentes.
    """
    x, y, indice = _matriz(grade_semanal(semanal), produto, defasagens)
    manter = np.ones(len(indice), dtype=bool)
    if inicio is not None:
        manter &= indice >= pd.Timestamp(inicio)
    if fim is not None:
        manter &= indice < pd.Timestamp(fim)
    x, y, indice = x[manter], y[manter], indice[manter]
    if len(y) < 3 * (defasagens + 2):
        raise ValueError(f"Poucas semanas ({len(y)}) para estimar {defasagens + 1} defasagens de {produto}.")

    betas = _betas(x, y)
    rng = np.random.default_rng(semente)
    inicios = np.arange(len(y) - bloco + 1)
    blocos = int(np.ceil(len(y) / bloco))
    sorteados = np.empty((n_boot, defasagens + 1))
    for i in range(n_boot):
        linhas = (rng.choice(inicios, size=blocos)[:, None] + np.arange(bloco)).ravel()[: len(y)]
        sorteados[i] = _betas(x[linhas], y[linhas])
    inferior, superior = np.percentile(np.cumsum(sorteados, axis=1), [5, 95], axis=0)
    return {
        "produto": produto,
        "semanas": int(len(y)),
        "periodo": [str(indice.min().date()), str(indice.max().date())],
        "acumulado": [round(float(v), 4) for v in np.cumsum(betas)],
        "ic90_inf": [round(float(v), 4) for v in inferior],
        "ic90_sup": [round(float(v), 4) for v in superior],
        "_betas": betas,
        "_betas_sorteados": sorteados,
    }


def _semanas_completas(grade: pd.DataFrame) -> pd.DataFrame:
    """Só as semanas em que o barril em reais existe (Brent e dólar com cotações suficientes)."""
    return grade[grade[BARRIL].notna()]


def _semana_base(grade: pd.DataFrame, produto: str) -> pd.Timestamp:
    """Última semana com barril E preço na bomba do produto: o "hoje" dos cenários.

    A bomba da ANP costuma sair antes do Brent da EIA, então normalmente é a
    última semana do barril. Se a bomba estiver mais de MAX_ATRASO_DA_BOMBA
    semanas atrás, o "preço de hoje" seria velho e parte do repasse já devido
    ficaria de fora: é melhor parar do que publicar o cenário.
    """
    completas = _semanas_completas(grade)
    if completas.empty:
        raise ValueError("A tabela semanal não tem nenhuma semana com barril em reais.")
    com_bomba = completas[completas[produto].notna()]
    if com_bomba.empty:
        raise ValueError(f"Nenhuma semana tem barril em reais e preço de {produto} ao mesmo tempo.")
    base, ultima = com_bomba.index[-1], completas.index[-1]
    atraso = int((ultima - base).days // 7)
    if atraso > MAX_ATRASO_DA_BOMBA:
        raise ValueError(
            f"O preço de {produto} na bomba para em {base.date()}, {atraso} semanas antes do barril "
            f"({ultima.date()}): atualize referencias/ antes de simular cenários."
        )
    return base


def situacao_atual(semanal: pd.DataFrame, produtos: list[str]) -> dict:
    """O ponto de partida dos cenários: barril, dólar e bomba da mesma semana.

    Traz também o dólar da semana mais recente com cotações suficientes, que
    pode ser posterior à do barril (a EIA publica o Brent com atraso).
    """
    grade = grade_semanal(semanal)
    bases = {produto: _semana_base(grade, produto) for produto in produtos}
    base = min(bases.values())
    dolar = grade["dolar"]
    if "dias_dolar" in grade:
        dolar = dolar.where(grade["dias_dolar"] >= MIN_DIAS_SEMANA)
    return {
        "semana": str(base.date()),
        "brent_usd": round(float(grade.at[base, "brent_usd"]), 2),
        "dolar": round(float(grade.at[base, "dolar"]), 4),
        "brent_reais_litro": round(float(grade.at[base, BARRIL]), 4),
        "bomba": {produto: round(float(grade.at[bases[produto], produto]), 3) for produto in produtos},
        "dolar_mais_recente": {
            "semana": str(dolar.last_valid_index().date()),
            "dolar": round(float(dolar.dropna().iloc[-1]), 4),
        },
    }


def hipoteses_do_historico(semanal: pd.DataFrame, janela: int = 52) -> dict[str, dict]:
    """Hipóteses de Brent e dólar tiradas da própria série, sem opinião embutida.

    Só contam semanas completas (com barril em reais). Pico e piso do Brent são
    as médias semanais extremas das últimas `janela` semanas; o dólar de
    estresse é a maior média semanal de toda a série. Mudam sozinhos quando os
    dados são atualizados.
    """
    completas = _semanas_completas(grade_semanal(semanal))
    if completas.empty:
        raise ValueError("A tabela semanal não tem nenhuma semana com barril em reais.")
    recente = completas.tail(janela)
    brent, dolar = float(completas["brent_usd"].iloc[-1]), float(completas["dolar"].iloc[-1])
    pico, piso = float(recente["brent_usd"].max()), float(recente["brent_usd"].min())
    dolar_maximo = float(completas["dolar"].max())
    return {
        "Barril e dólar ficam onde estão": {"brent_usd": brent, "dolar": dolar},
        "Escalada: Brent volta ao pico de 12 meses": {"brent_usd": pico, "dolar": dolar},
        "Escalada com dólar no máximo da série": {"brent_usd": pico, "dolar": dolar_maximo},
        "Alívio: Brent volta ao piso de 12 meses": {"brent_usd": piso, "dolar": dolar},
    }


def simular(semanal: pd.DataFrame, curva_: dict, hipotese: dict, horizonte: int = DEFASAGENS_REPASSE) -> pd.DataFrame:
    """Trajetória do preço na bomba de 0 a `horizonte` semanas depois do salto do barril.

    O barril salta para o nível da hipótese na semana seguinte à semana-base e
    fica lá. A linha `semanas_depois = k` é o preço k semanas depois do salto, e
    usa o repasse acumulado até a defasagem k, o mesmo da curva de repasse.

    Entra também o repasse ainda pendente das variações que o barril já teve (a
    regressão tem memória de `defasagens` semanas), então mesmo "tudo fica onde
    está" pode mexer o preço. É só o efeito do barril: a tendência média do
    preço (o intercepto da regressão) fica de fora.
    """
    grade = grade_semanal(semanal)
    produto, betas = curva_["produto"], curva_["_betas"]
    memoria = len(betas) - 1
    base = _semana_base(grade, produto)
    # buraco curto no barril (semana de feriado) é interpolado em log; sem isso as duas variações
    # que tocam a semana sumiriam e o repasse pendente delas viraria zero
    barril = np.exp(np.log(grade[BARRIL]).interpolate(limit=2, limit_area="inside")).loc[:base]
    historico = variacao_log(barril).to_numpy()[1:]
    # as últimas `memoria` variações já observadas (com zeros à esquerda se a série for curta);
    # o fatiamento [-memoria:] devolveria a série INTEIRA quando memoria é 0
    passado = historico[max(len(historico) - memoria, 0) :] if memoria else historico[:0]
    if np.isnan(passado).any():
        raise ValueError(
            f"O barril em reais tem um buraco de mais de 2 semanas nas {memoria} semanas antes de {base.date()}: "
            "não dá para calcular o repasse pendente."
        )
    passado = np.concatenate([np.zeros(memoria - len(passado)), passado])
    barril_novo = hipotese["brent_usd"] * hipotese["dolar"] / LITROS_POR_BARRIL
    passos = horizonte + 1
    futuro = np.zeros(passos)
    futuro[0] = 100 * np.log(barril_novo / grade.at[base, BARRIL])
    choques = np.concatenate([passado, futuro])

    def acumulado(coeficientes: np.ndarray) -> np.ndarray:
        # var_bomba(t) = soma_k b_k * var_barril(t-k); a semana do salto está na posição `memoria` de `choques`
        return np.cumsum(np.convolve(choques, coeficientes)[memoria : memoria + passos])

    preco_atual = float(grade.at[base, produto])
    central = acumulado(betas)
    sorteios = np.array([acumulado(b) for b in curva_["_betas_sorteados"]])
    inferior, superior = np.percentile(sorteios, [5, 95], axis=0)

    def preco(variacao: np.ndarray) -> np.ndarray:
        return preco_atual * np.exp(variacao / 100)

    return pd.DataFrame(
        {
            "semanas_depois": np.arange(passos),
            "variacao_pct": 100 * (np.exp(central / 100) - 1),
            "ic90_inf_pct": 100 * (np.exp(inferior / 100) - 1),
            "ic90_sup_pct": 100 * (np.exp(superior / 100) - 1),
            "preco": preco(central),
            "preco_ic90_inf": preco(inferior),
            "preco_ic90_sup": preco(superior),
        }
    )


def tabela_de_cenarios(
    semanal: pd.DataFrame, curvas: dict[str, dict], hipoteses: dict[str, dict], marcos: tuple[int, ...] = (4, 8, 16)
) -> pd.DataFrame:
    """Uma linha por cenário, curva e marco (semanas depois do salto), com preço e intervalo de 90%.

    `curvas` mapeia um rótulo (por exemplo "Gasolina comum, repasse de hoje")
    para o resultado de `curva`. A mesma hipótese pode ser lida em regimes diferentes.
    """
    grade = grade_semanal(semanal)
    linhas = []
    for cenario, hipotese in hipoteses.items():
        barril_novo = hipotese["brent_usd"] * hipotese["dolar"] / LITROS_POR_BARRIL
        for rotulo, curva_ in curvas.items():
            base = _semana_base(grade, curva_["produto"])
            trajetoria = simular(semanal, curva_, hipotese, horizonte=max(marcos)).set_index("semanas_depois")
            for marco in marcos:
                ponto = trajetoria.loc[marco]
                linhas.append(
                    {
                        "cenario": cenario,
                        "brent_usd": round(hipotese["brent_usd"], 1),
                        "dolar": round(hipotese["dolar"], 2),
                        "barril_reais_var_pct": round(100 * (barril_novo / float(grade.at[base, BARRIL]) - 1), 1),
                        "curva": rotulo,
                        "produto": curva_["produto"],
                        "semanas_depois": marco,
                        "preco_atual": round(float(grade.at[base, curva_["produto"]]), 3),
                        "variacao_pct": round(float(ponto["variacao_pct"]), 2),
                        "ic90_inf_pct": round(float(ponto["ic90_inf_pct"]), 2),
                        "ic90_sup_pct": round(float(ponto["ic90_sup_pct"]), 2),
                        "preco": round(float(ponto["preco"]), 3),
                        "preco_ic90_inf": round(float(ponto["preco_ic90_inf"]), 3),
                        "preco_ic90_sup": round(float(ponto["preco_ic90_sup"]), 3),
                    }
                )
    return pd.DataFrame(linhas)


def relatorio(
    semanal: pd.DataFrame, produtos: tuple[str, ...] = ("gasolina", "diesel_s10"), n_boot: int = 1000
) -> dict:
    """Tudo o que o notebook e o script mostram: curvas por período, situação atual e cenários.

    Os cenários usam o repasse do período atual. Para a gasolina entra também a
    leitura "se a paridade de importação voltasse", com a curva do período
    anterior: a mesma alta do barril dá resultados muito diferentes conforme a
    política de preços, e isso é parte do cenário.
    """
    curvas = {
        ROTULOS[produto]: {
            periodo: curva(semanal, produto, inicio=inicio, fim=fim, n_boot=n_boot)
            for periodo, (inicio, fim) in PERIODOS.items()
        }
        for produto in produtos
    }
    para_cenarios = {f"{rotulo}, repasse de hoje": periodos[DEPOIS_DO_CORTE] for rotulo, periodos in curvas.items()}
    if "gasolina" in produtos:
        para_cenarios[f"{ROTULOS['gasolina']}, se a paridade voltasse"] = curvas[ROTULOS["gasolina"]][ATE_O_CORTE]
    hipoteses = hipoteses_do_historico(semanal)
    return {
        "situacao_atual": situacao_atual(semanal, list(produtos)),
        "hipoteses": {nome: {k: round(v, 4) for k, v in h.items()} for nome, h in hipoteses.items()},
        "curvas": curvas,
        "cenarios": tabela_de_cenarios(semanal, para_cenarios, hipoteses),
    }


def resumo_das_curvas(curvas: dict[str, dict[str, dict]], marcos: tuple[int, ...] = (4, 8, 16)) -> dict:
    """Só os números de leitura (sem as matrizes do bootstrap), para gravar em JSON."""
    return {
        produto: {
            periodo: {
                "semanas": c["semanas"],
                "periodo": c["periodo"],
                "repasse_acumulado": {str(m): c["acumulado"][m] for m in marcos},
                "ic90": {str(m): [c["ic90_inf"][m], c["ic90_sup"][m]] for m in marcos},
            }
            for periodo, c in periodos.items()
        }
        for produto, periodos in curvas.items()
    }
