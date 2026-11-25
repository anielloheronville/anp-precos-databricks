import pytest

from anp_precos import bronze

CABECALHO = (
    "\ufeffRegiao - Sigla;Estado - Sigla;Municipio;Revenda;CNPJ da Revenda;Nome da Rua;Numero Rua;"
    "Complemento;Bairro;Cep;Produto;Data da Coleta;Valor de Venda;Valor de Compra;Unidade de Medida;Bandeira"
)


def test_le_csv_com_bom_aspas_e_quebra_de_linha(spark, tmp_path):
    linhas = [
        CABECALHO,
        # dados fictícios
        "SE;SP;CIDADE A;POSTO A; 99.888.777/0001-00;RUA X;1;;CENTRO;00000-000;"
        "ETANOL;03/01/2023;4,00;;R$ / litro;BRANCA",
        # complemento entre aspas com ';' e quebra de linha dentro: não pode deslocar colunas
        'CO;MT;CIDADE B;POSTO B;11.222.333/0001-81;AV Y;10;"LOJA 1;\nFUNDOS";CENTRO;00000-000;'
        "DIESEL S10;04/01/2023;6,21;;R$ / litro;IPIRANGA",
    ]
    arquivo = tmp_path / "amostra.csv"
    # bytes, para o Windows não trocar o \n de dentro das aspas por \r\n
    arquivo.write_bytes(("\r\n".join(linhas) + "\r\n").encode("utf-8"))

    df = bronze.ler_csv_anp(spark, str(arquivo))
    linhas_lidas = {r["revenda"]: r for r in df.collect()}

    assert df.count() == 2
    assert "regiao" in df.columns and "_arquivo" in df.columns
    assert linhas_lidas["POSTO B"]["complemento"] == "LOJA 1;\nFUNDOS"
    assert linhas_lidas["POSTO B"]["bandeira"] == "IPIRANGA"
    assert linhas_lidas["POSTO B"]["valor_venda"] == "6,21"


def test_cabecalho_diferente_falha_alto(spark, tmp_path):
    arquivo = tmp_path / "errado.csv"
    arquivo.write_text("A;B\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cabeçalho da ANP mudou"):
        bronze.ler_csv_anp(spark, str(arquivo))
