"""O script de download: conversão da planilha, conferência do conteúdo e janelas do Banco Central.

Nada aqui acessa a rede: as respostas das fontes são simuladas.
"""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

from anp_precos.esquema import COLUNAS_BOMBA
from test_planilha import PLANILHA_ANP, TITULOS, _xlsx  # o pytest põe a pasta tests/ no caminho de importação

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baixar():
    especificacao = importlib.util.spec_from_file_location("baixar", RAIZ / "scripts" / "baixar_referencias.py")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


def test_planilha_vira_o_csv_versionado_e_linha_sem_preco_fica_de_fora(baixar):
    csv_bytes, sem_preco = baixar.planilha_da_bomba_para_csv(PLANILHA_ANP)
    linhas = csv_bytes.decode("utf-8").splitlines()
    assert linhas[0] == ",".join(COLUNAS_BOMBA)
    assert linhas[1] == "2026-11-15,2026-11-21,GASOLINA COMUM,4584,R$/l,6.47"
    assert linhas[3] == "2026-11-15,2026-11-21,GNV,358,R$/m³,4.84"
    assert sem_preco == 0
    assert baixar.conferir_bomba(csv_bytes, minimo=3) == 3

    # a ANP marca com "-" o produto que não publicou naquela semana: a linha não entra, e é contada
    com_traco = _xlsx([TITULOS, [46341, 46347, "GASOLINA COMUM", 4584, "R$/l", 6.47],
                       [46341, 46347, "GLP", 3000, "R$/13kg", "-"]])
    csv_bytes, sem_preco = baixar.planilha_da_bomba_para_csv(com_traco)
    assert sem_preco == 1 and len(csv_bytes.decode("utf-8").splitlines()) == 2


@pytest.mark.parametrize(
    "linha",
    [
        [46342, 46348, "GASOLINA COMUM", 4584, "R$/l", 6.47],  # semana que não começa no domingo
        [46341, 46347, "QUEROSENE", 10, "R$/l", 5.0],  # produto fora do contrato
        [46341, 46347, "GASOLINA COMUM", 4584, "R$/l", 647],  # preço em centavos
        ["15/11/26", 46347, "GASOLINA COMUM", 4584, "R$/l", 6.47],  # data em texto que não é data
        ["Fonte: ANP", None, None, None, None, 1.0],  # rodapé
    ],
)
def test_conferencia_da_bomba_recusa_o_que_o_pipeline_recusaria(baixar, linha):
    csv_bytes, _ = baixar.planilha_da_bomba_para_csv(_xlsx([TITULOS, linha]))
    with pytest.raises(SystemExit, match="linha fora do contrato"):
        baixar.conferir_bomba(csv_bytes, minimo=1)


def _serie_bcb(n=30, extra=()) -> bytes:
    dias = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    linhas = ['"data";"valor"'] + [f'"{dia:%d/%m/%Y}";"5,{i:04d}"' for i, dia in enumerate(dias)]
    return ("\r\n".join([*linhas, *extra]) + "\r\n").encode("utf-8")


def test_conferencia_das_series_aceita_o_formato_real_e_recusa_conteudo_ruim(baixar, monkeypatch):
    monkeypatch.setitem(baixar.MINIMO_DE_LINHAS, "dolar", 10)
    assert baixar.conferir_serie("dolar", _serie_bcb()) == 30
    casos = {
        "parece cortada": _serie_bcb().rstrip(b"\r\n"),
        "cabeçalho inesperado": b"<!doctype html>\n<html></html>\n",
        "só 5 linhas": _serie_bcb(n=5),
        "valor que não é número": _serie_bcb(extra=['"01/03/2024";"n/d"']),
        "fora da faixa": _serie_bcb(extra=['"01/03/2024";"59,0"']),
        "data fora do formato": _serie_bcb(extra=['"2024-03-01";"5,0"']),
        "repetidas ou fora de ordem": _serie_bcb(extra=['"05/01/2024";"5,0"']),
        "sem exatamente 2 colunas": _serie_bcb(extra=['"01/03/2024";"5,0";"x"']),
    }
    for mensagem, conteudo in casos.items():
        with pytest.raises(SystemExit, match=mensagem):
            baixar.conferir_serie("dolar", conteudo)
    # dia sem cotação (valor vazio) não é erro, como no pipeline
    brent = b"observation_date,DCOILBRENTEU\n" + b"".join(
        f"2024-01-{d:02d},{'' if d == 5 else 80 + d}\n".encode() for d in range(1, 31)
    )
    monkeypatch.setitem(baixar.MINIMO_DE_LINHAS, "brent", 10)
    assert baixar.conferir_serie("brent", brent) == 30


def test_dolar_e_buscado_em_janelas_de_ate_dez_anos_e_emendado(baixar):
    inicio, fim = dt.date(2016, 12, 1), dt.date(2027, 6, 30)
    janelas = baixar.janelas(inicio, fim)
    assert len(janelas) == 2
    assert janelas[0][0] == inicio and janelas[-1][1] == fim
    assert all((ate - de).days < baixar.JANELA_BCB_DIARIA_DIAS for de, ate in janelas)
    assert janelas[1][0] == janelas[0][1] + dt.timedelta(days=1)  # sem buraco e sem sobreposição
    assert baixar.janelas(inicio, dt.date(2020, 1, 1)) == [(inicio, dt.date(2020, 1, 1))]

    pedidos = []

    def resposta_falsa(url: str) -> bytes:
        pedidos.append(url)
        dia = "01/12/2016" if len(pedidos) == 1 else "29/11/2026"
        return f'"data";"valor"\r\n"{dia}";"5,0000"\r\n'.encode()

    emendado = baixar.baixar_bcb_diaria(1, inicio, fim, buscar=resposta_falsa)
    assert len(pedidos) == 2 and "dataInicial=29%2F11%2F2026" in pedidos[1]
    assert emendado == b'"data";"valor"\r\n"01/12/2016";"5,0000"\r\n"29/11/2026";"5,0000"\r\n'  # um cabeçalho só
