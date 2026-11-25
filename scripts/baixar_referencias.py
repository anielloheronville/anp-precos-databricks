"""Baixa as séries de referência de fontes oficiais e grava em `referencias/`.

| arquivo                        | o que é                                   | fonte                                   |
|--------------------------------|-------------------------------------------|-----------------------------------------|
| brent_usd_diario.csv           | Brent spot, US$/barril, diário            | U.S. EIA, série DCOILBRENTEU do FRED    |
| dolar_ptax_venda_diario.csv    | dólar PTAX de venda, R$/US$, diário       | Banco Central, série 1 do SGS           |
| ipca_geral_mensal.csv          | IPCA, variação mensal em %                | IBGE, série 433 do SGS                  |
| ipca_alimentos_mensal.csv      | IPCA alimentação e bebidas, % ao mês      | IBGE, série 1635 do SGS                 |
| alimentos_mundo_mensal.csv     | índice mundial de alimentos em dólares    | FMI, série PFOODINDEXM do FRED          |
| bomba_brasil_semanal.csv       | preço médio de revenda no Brasil, semanal | ANP, série histórica do levantamento    |

Os cinco primeiros são gravados como vieram. O último é a planilha da ANP convertida em CSV (só
as colunas usadas; semana sem preço para um produto não entra), porque o Spark não lê .xlsx.
São arquivos pequenos e ficam versionados, para o pipeline rodar igual na máquina local, no CI
e no Databricks.

Tudo é baixado e conferido ANTES de qualquer gravação, com as mesmas regras do pipeline (formato
de data, número, faixa plausível, produto conhecido): se uma fonte falhar, nenhum arquivo
versionado é tocado.

O início padrão é fixo (dez/2016) para o estudo de eventos sempre alcançar 2017. O Banco Central
só aceita 10 anos por consulta em série diária, então o dólar é buscado em janelas e emendado.

Uso:
    python scripts/baixar_referencias.py
    python scripts/baixar_referencias.py --series ipca_geral bomba_brasil
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from anp_precos import planilha  # noqa: E402
from anp_precos.esquema import (  # noqa: E402
    ARQUIVOS_REFERENCIA,
    CABECALHO_PLANILHA_BOMBA,
    COLUNAS_BOMBA,
    FAIXA_PRECO_BOMBA,
    FONTES_REFERENCIA,
    PRODUTOS_BOMBA,
)

AGENTE = "anp-precos/0.1 (python-urllib)"
INICIO_PADRAO = dt.date(2016, 12, 1)
JANELA_BCB_DIARIA_DIAS = 3650  # acima disso o Banco Central responde HTTP 406
FORMATOS_PYTHON = {"yyyy-MM-dd": "%Y-%m-%d", "dd/MM/yyyy": "%d/%m/%Y"}
MINIMO_DE_LINHAS = {"brent": 100, "dolar": 100, "ipca_geral": 24, "ipca_alimentos": 24, "alimentos_mundo": 24}
URL_PLANILHA_ANP = (
    "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/"
    "precos-revenda-e-de-distribuicao-combustiveis/shlp/semanal/semanal-brasil-desde-2013.xlsx"
)


def url_fred(serie: str, inicio: dt.date, fim: dt.date) -> str:
    consulta = urllib.parse.urlencode({"id": serie, "cosd": inicio.isoformat(), "coed": fim.isoformat()})
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?{consulta}"


def url_bcb(serie: int, inicio: dt.date, fim: dt.date) -> str:
    consulta = urllib.parse.urlencode(
        {"formato": "csv", "dataInicial": inicio.strftime("%d/%m/%Y"), "dataFinal": fim.strftime("%d/%m/%Y")}
    )
    return f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?{consulta}"


def baixar(url: str) -> bytes:
    pedido = urllib.request.Request(url, headers={"User-Agent": AGENTE, "Accept": "*/*"})
    with urllib.request.urlopen(pedido, timeout=120) as resposta:
        return resposta.read()


def janelas(inicio: dt.date, fim: dt.date, dias: int = JANELA_BCB_DIARIA_DIAS) -> list[tuple[dt.date, dt.date]]:
    """Divide [inicio, fim] em janelas seguidas de no máximo `dias` dias, sem sobreposição."""
    saida, de = [], inicio
    while de <= fim:
        ate = min(de + dt.timedelta(days=dias - 1), fim)
        saida.append((de, ate))
        de = ate + dt.timedelta(days=1)
    return saida


def baixar_bcb_diaria(serie: int, inicio: dt.date, fim: dt.date, buscar=baixar) -> bytes:
    """Série diária do SGS em janelas de até 10 anos, emendadas: um cabeçalho só, linhas em ordem."""
    partes = [buscar(url_bcb(serie, de, ate)) for de, ate in janelas(inicio, fim)]
    corpo = partes[0] if partes[0].endswith(b"\n") else partes[0] + b"\n"
    for parte in partes[1:]:
        cabecalho, _, resto = parte.partition(b"\n")
        if cabecalho.strip() != partes[0].partition(b"\n")[0].strip():
            raise SystemExit(f"Série {serie} do SGS: janelas com cabeçalhos diferentes; resposta inesperada.")
        corpo += resto if resto.endswith(b"\n") or not resto else resto + b"\n"
    return corpo


def conferir_serie(fonte: str, conteudo: bytes) -> int:
    """Recusa o que o pipeline recusaria: página de erro, arquivo cortado, data ou valor fora do contrato."""
    config, nome = FONTES_REFERENCIA[fonte], ARQUIVOS_REFERENCIA[fonte]
    texto = conteudo.decode("utf-8-sig", errors="replace")
    if not texto.endswith("\n"):
        raise SystemExit(f"{nome}: a resposta não termina em quebra de linha; parece cortada")
    linhas = list(csv.reader(io.StringIO(texto), delimiter=config["sep"]))
    if not linhas or [c.strip() for c in linhas[0]] != list(config["colunas"]):
        raise SystemExit(f"{nome}: cabeçalho inesperado ({linhas[0] if linhas else '(vazio)'!r})")
    corpo = [linha for linha in linhas[1:] if linha]
    if len(corpo) < MINIMO_DE_LINHAS[fonte]:
        raise SystemExit(f"{nome}: só {len(corpo)} linhas; esperava pelo menos {MINIMO_DE_LINHAS[fonte]}")
    minimo, maximo = config["faixa"]
    anterior = None
    for linha in corpo:
        if len(linha) != 2:
            raise SystemExit(f"{nome}: linha sem exatamente 2 colunas: {linha!r}")
        try:
            data = dt.datetime.strptime(linha[0].strip(), FORMATOS_PYTHON[config["formato_data"]]).date()
        except ValueError:
            raise SystemExit(f"{nome}: data fora do formato da fonte: {linha!r}") from None
        if anterior is not None and data <= anterior:
            raise SystemExit(f"{nome}: datas repetidas ou fora de ordem em {linha!r}")
        anterior = data
        valor = linha[1].strip()
        if valor in ("", "."):  # dia sem cotação
            continue
        try:
            numero = float(valor.replace(",", "."))
        except ValueError:
            raise SystemExit(f"{nome}: valor que não é número: {linha!r}") from None
        if not math.isfinite(numero) or not minimo <= numero <= maximo:
            raise SystemExit(f"{nome}: valor fora da faixa {minimo} a {maximo}: {linha!r}")
    return len(corpo)


def _numero(valor) -> str:
    """Número da planilha como texto com ponto decimal; o que não é número vira vazio."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
        return ""
    return f"{valor:.4f}".rstrip("0").rstrip(".")


def planilha_da_bomba_para_csv(conteudo: bytes) -> tuple[bytes, int]:
    """Converte a planilha semanal da ANP em CSV, mantendo só as colunas do pipeline.

    Devolve o CSV e quantas linhas ficaram de fora por não terem preço numérico
    (semana em que a ANP não publicou aquele produto, marcada com "-").
    """
    saida = io.StringIO()
    escritor = csv.writer(saida, lineterminator="\n")
    escritor.writerow(COLUNAS_BOMBA)
    sem_preco = 0
    for registro in planilha.tabela(conteudo, CABECALHO_PLANILHA_BOMBA):
        preco = _numero(registro["preco_medio_revenda"])
        if not preco:
            sem_preco += 1
            continue
        inicio = planilha.serial_para_data(registro["data_inicial"])
        fim = planilha.serial_para_data(registro["data_final"])
        escritor.writerow(
            [
                inicio.isoformat() if inicio else str(registro["data_inicial"] or ""),
                fim.isoformat() if fim else str(registro["data_final"] or ""),
                str(registro["produto"] or "").strip(),
                _numero(registro["numero_de_postos"]),
                str(registro["unidade"] or "").strip(),
                preco,
            ]
        )
    return saida.getvalue().encode("utf-8"), sem_preco


def conferir_bomba(conteudo: bytes, minimo: int = 1000) -> int:
    """As mesmas regras de `referencias.bomba_invalidas`, antes de sobrescrever o arquivo versionado."""
    nome = ARQUIVOS_REFERENCIA["bomba_brasil"]
    linhas = list(csv.reader(io.StringIO(conteudo.decode("utf-8"))))
    if not linhas or linhas[0] != COLUNAS_BOMBA:
        raise SystemExit(f"{nome}: cabeçalho inesperado ({linhas[0] if linhas else '(vazio)'!r})")
    corpo = linhas[1:]
    if len(corpo) < minimo:
        raise SystemExit(f"{nome}: só {len(corpo)} linhas; esperava pelo menos {minimo}")
    preco_minimo, preco_maximo = FAIXA_PRECO_BOMBA
    for linha in corpo:
        try:
            inicio = dt.date.fromisoformat(linha[0])
            preco = float(linha[5])
            valida = len(linha) == len(COLUNAS_BOMBA) and inicio.weekday() == 6 and linha[2] in PRODUTOS_BOMBA
            valida = valida and preco_minimo <= preco <= preco_maximo
        except (ValueError, IndexError):
            valida = False
        if not valida:
            raise SystemExit(
                f"{nome}: linha fora do contrato (domingo ISO, produto conhecido, preço na faixa): {linha!r}"
            )
    return len(corpo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", type=dt.date.fromisoformat, default=INICIO_PADRAO)
    ap.add_argument("--fim", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--destino", default=str(RAIZ / "referencias"))
    ap.add_argument("--series", nargs="+", choices=sorted(ARQUIVOS_REFERENCIA), default=sorted(ARQUIVOS_REFERENCIA))
    args = ap.parse_args()
    inicio, fim = args.inicio, args.fim

    fontes = {
        "brent": lambda: baixar(url_fred("DCOILBRENTEU", inicio, fim)),
        "dolar": lambda: baixar_bcb_diaria(1, inicio, fim),
        "ipca_geral": lambda: baixar(url_bcb(433, inicio, fim)),
        "ipca_alimentos": lambda: baixar(url_bcb(1635, inicio, fim)),
        "alimentos_mundo": lambda: baixar(url_fred("PFOODINDEXM", inicio, fim)),
    }
    baixados = []
    for serie in args.series:
        if serie == "bomba_brasil":
            conteudo, sem_preco = planilha_da_bomba_para_csv(baixar(URL_PLANILHA_ANP))
            linhas = conferir_bomba(conteudo)
            nota = f" ({sem_preco} linhas da planilha sem preço ficaram de fora)" if sem_preco else ""
        else:
            conteudo = fontes[serie]()
            linhas, nota = conferir_serie(serie, conteudo), ""
        baixados.append((ARQUIVOS_REFERENCIA[serie], conteudo, linhas, nota))

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    for nome, conteudo, linhas, nota in baixados:
        temporario = destino / f".{nome}.tmp"
        temporario.write_bytes(conteudo)
        os.replace(temporario, destino / nome)
        print(f"{nome}: {linhas} linhas, {len(conteudo) / 1024:.0f} KB{nota}")


if __name__ == "__main__":
    main()
