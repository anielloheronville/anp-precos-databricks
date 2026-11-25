"""Baixa as séries de referência (Brent e dólar) de fontes oficiais, como vieram.

- Brent: preço spot Europe Brent FOB em US$/barril, diário. Fonte: U.S. EIA, pela
  série DCOILBRENTEU do FRED (CSV público, sem chave).
- Dólar: PTAX de venda em R$/US$, diária. Fonte: Banco Central do Brasil, série 1 do SGS.

Os arquivos são gravados sem nenhuma alteração; tipos e validação ficam no pipeline.
Eles são pequenos (cerca de 50 KB cada) e ficam versionados em `referencias/`, para o
pipeline rodar igual na máquina local, no CI e no Databricks.

As duas respostas são baixadas e conferidas ANTES de qualquer gravação: se uma falhar,
nenhum arquivo versionado é tocado.

Uso:
    python scripts/baixar_referencias.py                      # últimos 9 anos e meio
    python scripts/baixar_referencias.py --inicio 2022-07-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
AGENTE = "anp-precos/0.1 (python-urllib)"
# O Banco Central recusa (HTTP 406) consulta de série diária com janela maior que 10 anos.
JANELA_MAXIMA_DIAS = 3650
JANELA_PADRAO_DIAS = 3470
MINIMO_DE_LINHAS = 100


def url_brent(inicio: dt.date, fim: dt.date) -> str:
    consulta = urllib.parse.urlencode({"id": "DCOILBRENTEU", "cosd": inicio.isoformat(), "coed": fim.isoformat()})
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?{consulta}"


def url_dolar(inicio: dt.date, fim: dt.date) -> str:
    consulta = urllib.parse.urlencode(
        {"formato": "csv", "dataInicial": inicio.strftime("%d/%m/%Y"), "dataFinal": fim.strftime("%d/%m/%Y")}
    )
    return f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?{consulta}"


def baixar(url: str) -> bytes:
    pedido = urllib.request.Request(url, headers={"User-Agent": AGENTE, "Accept": "text/csv,*/*"})
    with urllib.request.urlopen(pedido, timeout=60) as resposta:
        return resposta.read()


def conferir(nome: str, conteudo: bytes, cabecalho: str, separador: str) -> int:
    """Recusa resposta que não é o CSV esperado (página de erro, arquivo cortado, série vazia)."""
    texto = conteudo.decode("utf-8-sig", errors="replace")
    linhas = texto.splitlines()
    if not linhas or linhas[0].replace('"', "").strip() != cabecalho:
        raise SystemExit(f"{nome}: cabeçalho inesperado ({linhas[0][:60]!r} em vez de {cabecalho!r})")
    if not texto.endswith("\n"):
        raise SystemExit(f"{nome}: a resposta não termina em quebra de linha; parece cortada")
    corpo = [linha for linha in linhas[1:] if linha.strip()]
    if len(corpo) < MINIMO_DE_LINHAS:
        raise SystemExit(f"{nome}: só {len(corpo)} linhas; esperava pelo menos {MINIMO_DE_LINHAS}")
    tortas = [linha for linha in corpo if linha.count(separador) != 1]
    if tortas:
        raise SystemExit(f"{nome}: {len(tortas)} linhas sem exatamente 2 colunas, por exemplo {tortas[0][:60]!r}")
    return len(corpo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", type=dt.date.fromisoformat, default=None, help="padrão: 9 anos e meio antes do fim")
    ap.add_argument("--fim", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--destino", default=str(RAIZ / "referencias"))
    args = ap.parse_args()

    inicio = args.inicio or args.fim - dt.timedelta(days=JANELA_PADRAO_DIAS)
    if (args.fim - inicio).days > JANELA_MAXIMA_DIAS:
        raise SystemExit(
            f"Janela de {(args.fim - inicio).days} dias: o Banco Central aceita no máximo 10 anos por consulta. "
            "Use um --inicio mais recente."
        )

    fontes = [
        ("brent_usd_diario.csv", url_brent(inicio, args.fim), "observation_date,DCOILBRENTEU", ","),
        ("dolar_ptax_venda_diario.csv", url_dolar(inicio, args.fim), "data;valor", ";"),
    ]
    baixados = []
    for nome, url, cabecalho, separador in fontes:
        conteudo = baixar(url)
        linhas = conferir(nome, conteudo, cabecalho, separador)
        baixados.append((nome, url, conteudo, linhas))

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    for nome, url, conteudo, linhas in baixados:
        temporario = destino / f".{nome}.tmp"
        temporario.write_bytes(conteudo)
        os.replace(temporario, destino / nome)
        print(f"{nome}: {linhas} linhas, {len(conteudo) / 1024:.0f} KB  <-  {url}")


if __name__ == "__main__":
    main()
