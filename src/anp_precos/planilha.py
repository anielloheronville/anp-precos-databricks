"""Leitor mínimo de .xlsx, só com a biblioteca padrão (zipfile + XML).

A ANP publica a série semanal de preços em planilha. Para não depender de um
pacote de Excel em três ambientes (local, CI e Databricks), o download converte
a planilha em CSV com este leitor, e o CSV é o que fica versionado.

Só lê valores: não interpreta fórmulas, macros nem formatação. Datas chegam
como número de série do Excel; use `serial_para_data`.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
EPOCA_EXCEL = dt.date(1899, 12, 30)


def _indice_da_coluna(referencia: str) -> int:
    letras = re.match(r"[A-Z]+", referencia).group(0)
    numero = 0
    for letra in letras:
        numero = numero * 26 + (ord(letra) - 64)
    return numero - 1


def _texto(elemento: ET.Element) -> str:
    return "".join(t.text or "" for t in elemento.iter(f"{NS}t"))


def linhas_da_primeira_planilha(conteudo: bytes) -> list[list]:
    """Células da primeira planilha, linha a linha: texto, número (float) ou None."""
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        pasta = ET.fromstring(z.read("xl/workbook.xml"))
        relacoes = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        alvo = {r.attrib["Id"]: r.attrib["Target"] for r in relacoes}
        primeira = pasta.find(f"{NS}sheets")[0]
        parte = alvo[primeira.attrib[REL]].lstrip("/")
        parte = parte if parte.startswith("xl/") else f"xl/{parte}"
        textos = []
        if "xl/sharedStrings.xml" in z.namelist():
            textos = [_texto(si) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{NS}si")]
        saida = []
        for linha in ET.fromstring(z.read(parte)).iter(f"{NS}row"):
            valores: list = []
            for celula in linha.findall(f"{NS}c"):
                # o atributo "r" (A1, B1...) é opcional no formato: sem ele, a célula é a próxima da linha
                indice = _indice_da_coluna(celula.attrib["r"]) if "r" in celula.attrib else len(valores)
                valores.extend([None] * (indice + 1 - len(valores)))
                tipo, valor = celula.attrib.get("t"), celula.find(f"{NS}v")
                if tipo == "inlineStr":
                    valores[indice] = _texto(celula)
                elif valor is None or valor.text is None:
                    valores[indice] = None
                elif tipo == "s":
                    valores[indice] = textos[int(valor.text)]
                elif tipo in ("str", "b", "e"):
                    valores[indice] = valor.text
                else:
                    try:
                        valores[indice] = float(valor.text)
                    except ValueError:
                        valores[indice] = valor.text
            saida.append(valores)
        return saida


def serial_para_data(valor) -> dt.date | None:
    """Número de série do Excel (ou texto dd/mm/aaaa) para data; None se não for uma data."""
    if isinstance(valor, (int, float)) and not isinstance(valor, bool) and 1 <= valor < 100000:
        return EPOCA_EXCEL + dt.timedelta(days=int(valor))
    if isinstance(valor, str):
        for formato in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(valor.strip(), formato).date()
            except ValueError:
                continue
    return None


def tabela(conteudo: bytes, cabecalho: dict[str, str]) -> list[dict]:
    """Linhas de dados da planilha como dicionários, só com as colunas pedidas.

    `cabecalho` mapeia o título da coluna na planilha para o nome de saída. A
    linha de cabeçalho é a primeira que contém todos os títulos; o que vem
    antes (título do relatório, observações) é ignorado.
    """
    linhas = linhas_da_primeira_planilha(conteudo)
    inicio, indices = None, {}
    for numero, linha in enumerate(linhas):
        titulos = [str(c).strip() if c is not None else "" for c in linha]
        if set(cabecalho) <= set(titulos):
            inicio = numero + 1
            indices = {saida: titulos.index(titulo) for titulo, saida in cabecalho.items()}
            break
    if inicio is None:
        raise ValueError(f"A planilha não tem uma linha de cabeçalho com as colunas {sorted(cabecalho)}")
    registros = []
    for linha in linhas[inicio:]:
        registro = {saida: (linha[i] if i < len(linha) else None) for saida, i in indices.items()}
        if any(v is not None and v != "" for v in registro.values()):
            registros.append(registro)
    return registros
