"""Leitor de .xlsx (sem Spark)."""

import datetime as dt
import io
import zipfile

import pytest

from anp_precos import planilha
from anp_precos.esquema import CABECALHO_PLANILHA_BOMBA

NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
REL = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _xlsx(linhas: list[list], com_posicao: bool = True) -> bytes:
    """Planilha mínima: textos em sharedStrings, números direto na célula, como o Excel grava."""
    textos: list[str] = []
    xml_linhas = []
    for n, linha in enumerate(linhas, start=1):
        celulas = []
        for c, valor in enumerate(linha):
            if valor is None:
                continue
            ref = f' r="{chr(65 + c)}{n}"' if com_posicao else ""
            if isinstance(valor, str):
                if valor not in textos:
                    textos.append(valor)
                celulas.append(f'<c{ref} t="s"><v>{textos.index(valor)}</v></c>')
            else:
                celulas.append(f'<c{ref}><v>{valor}</v></c>')
        xml_linhas.append(f'<row r="{n}">{"".join(celulas)}</row>')
    partes = {
        "xl/workbook.xml": (
            f'<workbook {NS} {REL}><sheets><sheet name="BRASIL" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="x" Target="worksheets/sheet1.xml"/></Relationships>'
        ),
        "xl/sharedStrings.xml": f"<sst {NS}>{''.join(f'<si><t>{t}</t></si>' for t in textos)}</sst>",
        "xl/worksheets/sheet1.xml": f"<worksheet {NS}><sheetData>{''.join(xml_linhas)}</sheetData></worksheet>",
    }
    saida = io.BytesIO()
    with zipfile.ZipFile(saida, "w") as z:
        for nome, conteudo in partes.items():
            z.writestr(nome, conteudo)
    return saida.getvalue()


TITULOS = list(CABECALHO_PLANILHA_BOMBA) + ["DESVIO PADRÃO REVENDA"]
PLANILHA_ANP = _xlsx(
    [
        ["AGÊNCIA NACIONAL DO PETRÓLEO"],
        ["Obs: não houve pesquisa entre 18/8/20 e 17/10/20."],
        [],
        TITULOS,
        # 46341 = domingo, 15/11/2026
        [46341, 46347, "GASOLINA COMUM", 4584, "R$/l", 6.47, 0.31],
        [46341, 46347, "OLEO DIESEL S10", 3189, "R$/l", 7.26, 0.386],
        [46341, 46347, "GNV", 358, "R$/m³", 4.84, "-"],
    ]
)


def test_numero_de_serie_do_excel_vira_data():
    assert planilha.serial_para_data(46341) == dt.date(2026, 11, 15)
    assert planilha.serial_para_data("15/11/2026") == dt.date(2026, 11, 15)
    assert planilha.serial_para_data("GASOLINA") is None
    assert planilha.serial_para_data(None) is None


def test_tabela_acha_o_cabecalho_depois_das_observacoes_e_ignora_colunas_a_mais():
    registros = planilha.tabela(PLANILHA_ANP, CABECALHO_PLANILHA_BOMBA)
    assert len(registros) == 3
    assert registros[0] == {
        "data_inicial": 46341.0,
        "data_final": 46347.0,
        "produto": "GASOLINA COMUM",
        "numero_de_postos": 4584.0,
        "unidade": "R$/l",
        "preco_medio_revenda": 6.47,
    }


def test_planilha_sem_o_cabecalho_esperado_falha_alto():
    with pytest.raises(ValueError, match="não tem uma linha de cabeçalho"):
        planilha.tabela(_xlsx([["A", "B"], [1, 2]]), CABECALHO_PLANILHA_BOMBA)


def test_linha_parcial_e_celula_sem_posicao_nao_somem_nem_quebram():
    conteudo = _xlsx([TITULOS, [46341, None, "GLP"], [46341, 46347, "GNV", 358, "R$/m³", "-"]])
    registros = planilha.tabela(conteudo, CABECALHO_PLANILHA_BOMBA)
    assert len(registros) == 2  # a linha incompleta continua lá, para a conferência recusar
    assert registros[0]["data_final"] is None and registros[0]["preco_medio_revenda"] is None
    assert registros[1]["preco_medio_revenda"] == "-"
    # célula sem o atributo "r" (opcional no formato) entra na coluna seguinte
    sem_posicao = _xlsx([TITULOS, [46341, 46347, "GNV", 358, "R$/m³", 4.84]], com_posicao=False)
    assert planilha.tabela(sem_posicao, CABECALHO_PLANILHA_BOMBA) == [
        {"data_inicial": 46341.0, "data_final": 46347.0, "produto": "GNV", "numero_de_postos": 358.0,
         "unidade": "R$/m³", "preco_medio_revenda": 4.84}
    ]
