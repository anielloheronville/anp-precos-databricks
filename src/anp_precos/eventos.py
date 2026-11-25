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
PERIODO_ANTES, PERIODO_DO_CHOQUE = list(PERIODOS)[:2]
ESTIMATIVAS = ("sem_controle_de_periodo", "com_controle_de_periodo")
# A fração acompanhada pelos canais é recalculada com estes números de defasagens e com o período do
# choque terminando nestes meses, para mostrar o quanto o resultado depende de escolhas de quem modela.
SENSIBILIDADE_DEFASAGENS = (1, 2, 3, 4, 6, 8)
SENSIBILIDADE_FIM_DO_CHOQUE = ("2022-06", "2022-12", "2023-06")
# Com menos meses que isso em um dos dois períodos, a diferença de médias não diz nada.
MIN_MESES_POR_PERIODO = 12


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


def _dentro(indice: pd.PeriodIndex, de: str | None, ate: str | None) -> np.ndarray:
    """Máscara dos meses de um período (as duas pontas incluídas; None = sem limite)."""
    dentro = np.ones(len(indice), dtype=bool)
    if de:
        dentro &= indice >= pd.Period(de, freq="M")
    if ate:
        dentro &= indice <= pd.Period(ate, freq="M")
    return dentro


def _regressao(dados: pd.DataFrame, defasagens: int) -> dict:
    """IPCA de alimentos nas variações dos canais, do mês e dos `defasagens` anteriores."""
    niveis = dados[[NIVEIS[canal] for canal in CANAIS]]
    interpolado = np.exp(np.log(niveis).interpolate(limit=1, limit_area="inside"))
    variacoes = pd.DataFrame(
        {canal: 100 * np.log(interpolado[NIVEIS[canal]] / interpolado[NIVEIS[canal]].shift(1)) for canal in CANAIS}
    )
    alvo = dados["ipca_alimentos"]
    x = pd.concat({(canal, k): variacoes[canal].shift(k) for canal in CANAIS for k in range(defasagens + 1)}, axis=1)
    completos = alvo.notna() & x.notna().all(axis=1)
    matriz, y = x[completos].to_numpy(), alvo[completos].to_numpy()
    coeficientes, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), matriz]), y, rcond=None)
    return {
        "x": matriz,
        "y": y,
        "indice": alvo[completos].index,
        "intercepto": float(coeficientes[0]),
        "betas": coeficientes[1:],
        "interpolados": int((niveis.isna() & interpolado.notna()).any(axis=1).sum()),
    }


def _periodos_com_fim(fim_do_choque: str) -> dict[str, tuple]:
    """Os mesmos três períodos, com o do choque terminando em outro mês."""
    antes, choque, depois = list(PERIODOS)
    seguinte = str(pd.Period(fim_do_choque, freq="M") + 1)
    return {antes: PERIODOS[antes], choque: (PERIODOS[choque][0], fim_do_choque), depois: (seguinte, None)}


def _betas(x: np.ndarray, y: np.ndarray, controles: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Coeficientes dos canais e, se houver, dos controles de período (o intercepto fica de fora)."""
    colunas = [np.ones(len(y)), x] if controles is None else [np.ones(len(y)), controles, x]
    coeficientes, *_ = np.linalg.lstsq(np.column_stack(colunas), y, rcond=None)
    extras = 0 if controles is None else controles.shape[1]
    return coeficientes[1 + extras :], coeficientes[1 : 1 + extras]


def _fracao(x: np.ndarray, betas: np.ndarray, antes: np.ndarray, choque: np.ndarray, aceleracao: float) -> float:
    """Parte da aceleração dos alimentos que a contribuição dos canais acompanha, com estes coeficientes."""
    contribuicao = x @ betas
    return float((contribuicao[choque].mean() - contribuicao[antes].mean()) / aceleracao)


def _controles(mascaras: list[np.ndarray]) -> np.ndarray:
    """Uma coluna 0/1 para cada período depois do primeiro que tenha meses: o nível médio de cada período."""
    return np.column_stack([m.astype(float) for m in mascaras[1:] if m.any()])


def _conta(x: np.ndarray, y: np.ndarray, mascaras: list[np.ndarray]) -> dict:
    """As duas frações para estas linhas, sem conferir se há meses suficientes (ver `_estimativas`).

    Numerador e denominador usam os MESMOS meses. Assim, na conta com controle,
    fração x aceleração + salto sem explicação = aceleração.
    """
    antes, choque = mascaras[0], mascaras[1]
    aceleracao = float(y[choque].mean() - y[antes].mean())
    controles = _controles(mascaras)
    simples, _ = _betas(x, y)
    controlado, saltos = _betas(x, y, controles)
    return {
        "antes": antes,
        "choque": choque,
        "aceleracao": aceleracao,
        "controles": controles,
        "betas": dict(zip(ESTIMATIVAS, (simples, controlado), strict=True)),
        "fracoes": {
            nome: _fracao(x, betas, antes, choque, aceleracao)
            for nome, betas in zip(ESTIMATIVAS, (simples, controlado), strict=True)
        },
        "salto": float(saltos[0]),
    }


def _estimativas(ajuste: dict, periodos: dict[str, tuple]) -> dict | None:
    """As duas estimativas da fração para uma especificação (defasagens e corte dos períodos).

    Não há estimativa quando falta base para a conta:
    - menos de `MIN_MESES_POR_PERIODO` meses em um dos dois períodos;
    - menos de dois meses por coeficiente a estimar;
    - aceleração que não passa de duas vezes o erro-padrão da diferença de
      médias (ou negativa): a fração seria uma divisão por quase nada.
    """
    x, y = ajuste["x"], ajuste["y"]
    mascaras = [_dentro(ajuste["indice"], *limites) for limites in periodos.values()]
    antes, choque = mascaras[0], mascaras[1]
    if antes.sum() < MIN_MESES_POR_PERIODO or choque.sum() < MIN_MESES_POR_PERIODO:
        return None
    if len(y) < 2 * (1 + _controles(mascaras).shape[1] + x.shape[1]):
        return None
    aceleracao = y[choque].mean() - y[antes].mean()
    erro = np.sqrt(y[choque].var(ddof=1) / choque.sum() + y[antes].var(ddof=1) / antes.sum())
    if not aceleracao > 2 * erro:
        return None
    return _conta(x, y, mascaras) | {"mascaras": mascaras}


def _reamostra(rng: np.random.Generator, n: int, bloco: int) -> np.ndarray:
    """`n` posições de 0 a n-1: blocos de `bloco` posições seguidas, cada um começando em um ponto sorteado."""
    tamanho = min(bloco, n)
    inicios = rng.integers(0, n - tamanho + 1, size=int(np.ceil(n / tamanho)))
    return (inicios[:, None] + np.arange(tamanho)).ravel()[:n]


def _sorteios_dos_coeficientes(ajuste: dict, estimativa: dict, n_boot: int, bloco: int, semente: int) -> np.ndarray:
    """Frações com os coeficientes reajustados em reamostras em blocos (colunas: sem e com controle).

    Só os coeficientes são sorteados. Os canais e a aceleração dos alimentos
    ficam como foram observados: o intervalo que sai daqui responde "e se os
    coeficientes fossem outros?", e é mais estreito que a incerteza da conta.
    """
    x, y, controles = ajuste["x"], ajuste["y"], estimativa["controles"]
    antes, choque, aceleracao = estimativa["antes"], estimativa["choque"], estimativa["aceleracao"]
    rng = np.random.default_rng(semente)
    saida = np.empty((n_boot, 2))
    for i in range(n_boot):
        linhas = _reamostra(rng, len(y), bloco)
        saida[i, 0] = _fracao(x, _betas(x[linhas], y[linhas])[0], antes, choque, aceleracao)
        saida[i, 1] = _fracao(x, _betas(x[linhas], y[linhas], controles[linhas])[0], antes, choque, aceleracao)
    return saida


def _sorteios_da_conta_inteira(ajuste: dict, estimativa: dict, n_boot: int, bloco: int, semente: int) -> np.ndarray:
    """Frações com a conta toda refeita em reamostras em blocos (colunas: sem e com controle).

    Os meses são sorteados dentro de cada período, que mantém o seu tamanho, e
    tudo é recalculado com eles: coeficientes, contribuição dos canais e
    aceleração dos alimentos. É a incerteza de toda a conta para uma
    especificação, e o intervalo que deve acompanhar a fração.
    """
    x, y = ajuste["x"], ajuste["y"]
    posicoes = [np.flatnonzero(m) for m in estimativa["mascaras"] if m.any()]
    periodo = np.concatenate([np.full(len(p), i) for i, p in enumerate(posicoes)])
    rng = np.random.default_rng(semente)
    saida = np.empty((n_boot, 2))
    for i in range(n_boot):
        linhas = np.concatenate([p[_reamostra(rng, len(p), bloco)] for p in posicoes])
        fracoes = _conta(x[linhas], y[linhas], [periodo == j for j in range(len(posicoes))])["fracoes"]
        saida[i] = [fracoes[nome] for nome in ESTIMATIVAS]
    return saida


def _por_canal(ajuste: dict, estimativa: dict, nome: str, passo: int) -> dict:
    betas, saida = estimativa["betas"][nome], {}
    for i, canal in enumerate(CANAIS):
        so_este = np.zeros_like(betas)
        so_este[i * passo : (i + 1) * passo] = betas[i * passo : (i + 1) * passo]
        fracao = _fracao(ajuste["x"], so_este, estimativa["antes"], estimativa["choque"], estimativa["aceleracao"])
        saida[canal] = round(fracao, 2)
    return saida


def _grade(dados: pd.DataFrame, ajuste: dict, defasagens: int) -> dict:
    """As duas frações, sem arredondar, em cada combinação de defasagens e de fim do período do choque.

    As defasagens da conta principal entram na grade mesmo que não estejam na
    lista. Fica None a combinação sem base para a conta (ver `_estimativas`) e
    a que repetiria a anterior: se a tabela acaba antes de um corte, os cortes
    seguintes dariam a mesma conta e não são combinações diferentes.
    """
    grade = {}
    for k in sorted({*SENSIBILIDADE_DEFASAGENS, defasagens}):
        deste = ajuste if k == defasagens else _regressao(dados, k)
        ultimo, acabou, linha = deste["indice"].max(), False, {}
        for fim in SENSIBILIDADE_FIM_DO_CHOQUE:
            estimativa = None if acabou else _estimativas(deste, _periodos_com_fim(fim))
            linha[fim] = None if estimativa is None else estimativa["fracoes"]
            acabou = acabou or bool(pd.Period(fim, freq="M") >= ultimo)
        grade[str(k)] = linha
    return grade


def _faixa(grade: dict, nome: str) -> dict | None:
    """Mínimo, mediana e máximo de uma das estimativas nas combinações da grade que têm conta.

    Recebe a grade sem arredondar: a mediana de valores já arredondados pode
    cair um ponto ao lado.
    """
    valores = [celula[nome] for linha in grade.values() for celula in linha.values() if celula is not None]
    if not valores:
        return None
    return {
        "combinacoes": len(valores),
        "minimo": round(min(valores), 2),
        "mediana": round(float(np.median(valores)), 2),
        "maximo": round(max(valores), 2),
        "com_metade_ou_mais": sum(valor >= 0.5 for valor in valores),
    }


def _arredondada(grade: dict) -> dict:
    return {
        k: {fim: celula and {nome: round(valor, 2) for nome, valor in celula.items()} for fim, celula in linha.items()}
        for k, linha in grade.items()
    }


def _aceleracao(
    dados: pd.DataFrame, ajuste: dict, defasagens: int, n_boot: int, bloco: int, semente: int
) -> dict | None:
    """Quanto da aceleração entre os dois períodos os canais acompanham, e o quanto esse número é frágil.

    A aceleração é a diferença da inflação média de alimentos entre o período
    do choque e o anterior. A fração é a diferença da contribuição média dos
    canais dividida por ela, nos meses que entram na regressão.

    Saem duas estimativas, porque a regressão simples tem um viés conhecido: se
    os alimentos deram um salto que os canais não explicam, ela atribui parte
    dele a quem subiu no mesmo período. A segunda dá a cada período o seu
    próprio nível médio e tira os coeficientes só do que varia DENTRO dos
    períodos; o que os canais não acompanham aparece como salto sem explicação.
    É a estimativa em que se deve apoiar.

    O ponto sozinho engana. Por isso saem também:
    - o intervalo de 90% da conta inteira (ver `_sorteios_da_conta_inteira`);
    - o intervalo de 90% só dos coeficientes, mais estreito (ver
      `_sorteios_dos_coeficientes`);
    - a fração em cada combinação de número de defasagens e de mês em que o
      período do choque termina (ver `_grade`), com mínimo, mediana e máximo.
    As pontas dos intervalos mexem alguns pontos com a semente e o tamanho do
    bloco. Defasagens e data de corte são escolhas de quem modela, e há outras
    que a grade não cobre (início da série, início do choque, moeda dos
    alimentos no mundo). A leitura honesta é a da faixa inteira, não a do ponto.
    """
    principal = _estimativas(ajuste, PERIODOS)
    if principal is None:
        return None
    alvo = dados["ipca_alimentos"]
    todos = [alvo[_dentro(dados.index, *PERIODOS[nome]) & alvo.notna().to_numpy()].mean()
             for nome in (PERIODO_ANTES, PERIODO_DO_CHOQUE)]
    intervalos = {
        "ic90_da_conta_inteira": _sorteios_da_conta_inteira(ajuste, principal, n_boot, bloco, semente),
        "ic90_dos_coeficientes": _sorteios_dos_coeficientes(ajuste, principal, n_boot, bloco, semente),
    }
    intervalos = {chave: np.percentile(sorteios, [5, 95], axis=0) for chave, sorteios in intervalos.items()}
    passo = defasagens + 1
    saida = {
        "de": PERIODO_ANTES,
        "para": PERIODO_DO_CHOQUE,
        "alimentos_pontos_ao_mes": round(float(todos[1] - todos[0]), 2),
        "alimentos_pontos_ao_mes_nos_meses_da_regressao": round(principal["aceleracao"], 2),
    }
    for coluna, nome in enumerate(ESTIMATIVAS):
        saida[nome] = {"fracao_acompanhada": round(principal["fracoes"][nome], 2)}
        for chave, limites in intervalos.items():
            saida[nome][chave] = [round(float(v), 2) + 0.0 for v in limites[:, coluna]]  # + 0.0: sem -0.0
        saida[nome]["por_canal"] = _por_canal(ajuste, principal, nome, passo)
    saida["com_controle_de_periodo"]["salto_sem_explicacao_pontos_ao_mes"] = round(principal["salto"], 2)
    grade = _grade(dados, ajuste, defasagens)
    saida["por_defasagens_e_fim_do_choque"] = _arredondada(grade)
    saida["faixa_das_especificacoes"] = {nome: _faixa(grade, nome) for nome in ESTIMATIVAS}
    return saida


def canais(mensal: pd.DataFrame, defasagens: int = 4, n_boot: int = 4000, bloco: int = 6, semente: int = 42) -> dict:
    """Quanto da inflação de alimentos os canais medidos acompanham, no todo e em cada período.

    Regressão do IPCA de alimentos nas variações, do mês e dos `defasagens`
    anteriores, dos alimentos no mundo em reais e do diesel na bomba.

    A ANP não pesquisou preços em setembro de 2020. Sem tratamento, esse único
    mês sem diesel tiraria da conta setembro de 2020 a fevereiro de 2021,
    justamente os meses de maior inflação de alimentos da série. Um buraco de
    um mês no canal é interpolado (em log) entre os vizinhos; buraco maior não.

    Por período: a inflação de alimentos de TODOS os meses do período, e a
    contribuição média dos dois canais (sem o intercepto) nos meses que entram
    na regressão. O intercepto é a inflação de alimentos que não depende deles.
    `aceleracao` resume quanto da alta entre os dois primeiros períodos os
    canais acompanham, com intervalos e sensibilidade (ver `_aceleracao`).
    """
    if n_boot < 1 or bloco < 1:
        raise ValueError(f"n_boot e bloco precisam ser pelo menos 1 (vieram {n_boot} e {bloco})")
    dados = preparar(mensal)
    ajuste = _regressao(dados, defasagens)
    x, y, indice = ajuste["x"], ajuste["y"], ajuste["indice"]
    observados, regressores = len(y), x.shape[1]
    contribuicao = x @ ajuste["betas"]
    residuo = y - ajuste["intercepto"] - contribuicao
    r2 = 1 - float((residuo**2).sum()) / float(((y - y.mean()) ** 2).sum())

    alvo = dados["ipca_alimentos"]
    por_periodo = {}
    for nome, (de, ate) in PERIODOS.items():
        todos = _dentro(dados.index, de, ate) & alvo.notna().to_numpy()
        na_regressao = _dentro(indice, de, ate)
        if not todos.any():
            continue
        media = float(alvo[todos].mean())
        por_periodo[nome] = {
            "meses": int(todos.sum()),
            "alimentos_pct_ao_mes": round(media, 2),
            "alimentos_pct_ao_ano": round(((1 + media / 100) ** 12 - 1) * 100, 1),
            "meses_na_regressao": int(na_regressao.sum()),
            "contribuicao_dos_canais_pct_ao_mes": (
                round(float(contribuicao[na_regressao].mean()), 2) if na_regressao.any() else None
            ),
        }
    passo = defasagens + 1
    return {
        "meses": observados,
        "meses_com_canal_interpolado": ajuste["interpolados"],
        "defasagens": defasagens,
        "r2": round(r2, 3),
        "r2_ajustado": round(1 - (1 - r2) * (observados - 1) / (observados - regressores - 1), 3),
        "intercepto_pct_ao_mes": round(ajuste["intercepto"], 2),
        "soma_dos_coeficientes": {
            canal: round(float(ajuste["betas"][i * passo : (i + 1) * passo].sum()), 3) for i, canal in enumerate(CANAIS)
        },
        "por_periodo": por_periodo,
        "aceleracao": _aceleracao(dados, ajuste, defasagens, n_boot, bloco, semente),
    }
