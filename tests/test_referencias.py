import csv
import datetime as dt
from pathlib import Path

import pytest

from anp_precos import referencias
from anp_precos.esquema import (
    ARQUIVOS_REFERENCIA,
    COLUNAS_BOMBA,
    FAIXA_PRECO_BOMBA,
    FONTES_REFERENCIA,
    LITROS_POR_BARRIL,
    PRODUTOS_BOMBA,
)

BRONZE = "data string, valor string"
DIARIO = "data date, brent_usd double, dolar double"
RAIZ = Path(__file__).resolve().parents[1]


def _arquivos(tmp_path, brent: list[str], dolar: list[str]) -> tuple[str, str]:
    # formatos reais: FRED com vírgula e data ISO; Banco Central com aspas, ';' e vírgula decimal
    arq_brent = tmp_path / "brent.csv"
    arq_brent.write_bytes(("\n".join(["observation_date,DCOILBRENTEU", *brent]) + "\n").encode("utf-8"))
    arq_dolar = tmp_path / "dolar.csv"
    arq_dolar.write_bytes(("\r\n".join(['"data";"valor"', *dolar]) + "\r\n").encode("utf-8"))
    return str(arq_brent), str(arq_dolar)


def test_le_os_dois_formatos_e_junta_por_dia(spark, tmp_path):
    arq_brent, arq_dolar = _arquivos(
        tmp_path,
        brent=["2026-03-09,118.40", "2026-03-10,", "2026-03-11,112.00"],  # dia 10: feriado, valor vazio
        dolar=['"09/03/2026";"5,2000"', '"10/03/2026";"5,2500"'],
    )
    bronze_brent = referencias.ler(spark, arq_brent, "brent")
    bronze_dolar = referencias.ler(spark, arq_dolar, "dolar")
    assert bronze_brent.columns == ["data", "valor", "_arquivo", "_ingestao_em"]
    assert bronze_dolar.first()["valor"] == "5,2000"  # bronze guarda o texto como veio

    dias = {r["data"]: r for r in referencias.diario(bronze_brent, bronze_dolar).collect()}
    assert dias[dt.date(2026, 3, 9)]["brent_usd"] == pytest.approx(118.40)
    assert dias[dt.date(2026, 3, 9)]["dolar"] == pytest.approx(5.20)
    assert dias[dt.date(2026, 3, 10)]["brent_usd"] is None  # feriado lá fora, dólar cotado aqui
    assert dias[dt.date(2026, 3, 10)]["dolar"] == pytest.approx(5.25)
    assert dias[dt.date(2026, 3, 11)]["dolar"] is None
    # feriado não é erro
    assert referencias.invalidas(bronze_brent, "brent").count() == 0
    assert referencias.invalidas(bronze_dolar, "dolar").count() == 0


def test_cabecalho_diferente_falha_alto(spark, tmp_path):
    arquivo = tmp_path / "errado.csv"
    arquivo.write_text("dia,preco\n2026-03-09,118.40\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cabeçalho da série 'brent' mudou"):
        referencias.ler(spark, str(arquivo), "brent")


@pytest.mark.parametrize("linha", ["2026-03-10,1,234.56", "2026-03-10"])  # coluna a mais; coluna a menos
def test_linha_com_numero_errado_de_colunas_derruba_a_leitura(spark, tmp_path, linha):
    arquivo = tmp_path / "torto.csv"
    arquivo.write_text(f"observation_date,DCOILBRENTEU\n2026-03-09,118.40\n{linha}\n", encoding="utf-8")
    with pytest.raises(Exception, match="(?i)malformed|FAILFAST"):
        referencias.ler(spark, str(arquivo), "brent").collect()


def test_linha_que_nao_e_data_e_numero_plausivel_e_denunciada(spark):
    bronze = spark.createDataFrame(
        [
            ("2026-03-09", "118.40"),
            ("2026-03-10", ""),  # sem cotação: aceito
            ("2026-03-11", "."),  # sem cotação no formato antigo do FRED: aceito
            ("2026-03-12", "n/d"),  # texto que não é número
            ("2026-03-13", "-3.5"),  # preço negativo
            ("2026-03-16", "NaN"),  # no Spark, NaN é "maior que tudo": passaria num simples valor > 0
            ("2026-03-17", "Infinity"),
            ("2026-03-18", "0.8"),  # número, mas fora da faixa do Brent (arquivo cortado em "80.0")
            ("12/03/2026", "110.00"),  # data fora do formato da fonte
        ],
        BRONZE,
    )
    ruins = {(r["data_texto"], r["valor_texto"]) for r in referencias.invalidas(bronze, "brent").collect()}
    assert ruins == {
        ("2026-03-12", "n/d"),
        ("2026-03-13", "-3.5"),
        ("2026-03-16", "NaN"),
        ("2026-03-17", "Infinity"),
        ("2026-03-18", "0.8"),
        ("12/03/2026", "110.00"),
    }
    # e nenhuma delas entra na série diária
    dolar = spark.createDataFrame([("09/03/2026", "5,00")], BRONZE)
    dias = {r["data"]: r["brent_usd"] for r in referencias.diario(bronze, dolar).collect()}
    assert dias == {dt.date(2026, 3, 9): pytest.approx(118.40)}


def test_mesmo_dia_repetido_conta_uma_vez_pela_media(spark):
    brent = spark.createDataFrame([("2026-03-09", "80.0"), ("2026-03-09", "100.0")], BRONZE)
    dolar = spark.createDataFrame([("09/03/2026", "5,00")], BRONZE)
    dias = referencias.diario(brent, dolar).collect()
    assert len(dias) == 1
    assert dias[0]["brent_usd"] == pytest.approx(90.0)


def _semana(segunda: dt.date, brent: float | None, dolar: float | None, dias: int = 5) -> list[tuple]:
    return [(segunda + dt.timedelta(days=i), brent, dolar) for i in range(dias)]


def _semanal(spark, linhas) -> dict:
    return {r["semana"]: r for r in referencias.brent_semanal(spark.createDataFrame(linhas, DIARIO)).collect()}


def test_barril_em_reais_por_litro_e_variacao_semanal(spark):
    s0 = dt.date(2026, 3, 2)  # segunda-feira
    semanas = _semanal(spark, _semana(s0, 80.0, 5.0) + _semana(s0 + dt.timedelta(days=7), 100.0, 5.5))

    primeira, segunda = semanas[s0], semanas[s0 + dt.timedelta(days=7)]
    assert primeira["brent_reais_litro"] == pytest.approx(80.0 * 5.0 / LITROS_POR_BARRIL)
    assert primeira["var_brent_1s"] is None  # não há semana anterior
    assert segunda["var_brent_1s"] == pytest.approx((100.0 * 5.5) / (80.0 * 5.0) - 1)
    assert segunda["var_brent_4s"] is None and segunda["var_brent_8s"] is None


def test_semana_incompleta_fica_sem_preco(spark):
    s = [dt.date(2026, 3, 2) + dt.timedelta(days=7 * i) for i in range(5)]
    linhas = (
        _semana(s[0], 80.0, 5.0)
        + _semana(s[1], 200.0, 5.0, dias=2)  # só 2 cotações de cada série
        + _semana(s[2], 90.0, 5.0)
        # Brent a semana inteira, dólar só 2 dias: basta UMA série incompleta para não haver preço
        + [(s[3] + dt.timedelta(days=i), 95.0, 5.0 if i < 2 else None) for i in range(5)]
        + _semana(s[4], 99.0, 5.0, dias=3)  # exatamente 3 cotações: é o mínimo, tem preço
    )
    semanas = _semanal(spark, linhas)
    assert semanas[s[1]]["brent_reais_litro"] is None
    assert semanas[s[1]]["var_brent_1s"] is None
    assert semanas[s[2]]["var_brent_1s"] is None  # a anterior não tem preço
    assert (semanas[s[3]]["dias_brent"], semanas[s[3]]["dias_dolar"]) == (5, 2)
    assert semanas[s[3]]["brent_reais_litro"] is None
    assert semanas[s[4]]["brent_reais_litro"] == pytest.approx(99.0 * 5.0 / LITROS_POR_BARRIL)


def test_variacao_usa_a_semana_exata_e_nao_a_linha_anterior(spark):
    s0 = dt.date(2026, 1, 5)
    # semanas 0, 1, 2 e 4: a semana 3 não existe na série
    linhas = []
    for i, preco in [(0, 80.0), (1, 84.0), (2, 88.0), (4, 100.0)]:
        linhas += _semana(s0 + dt.timedelta(days=7 * i), preco, 5.0)
    quinta = _semanal(spark, linhas)[s0 + dt.timedelta(days=28)]
    assert quinta["var_brent_1s"] is None  # não compara com 2 semanas atrás fingindo ser 1
    assert quinta["var_brent_4s"] == pytest.approx(100.0 / 80.0 - 1)  # a semana 0 existe, 28 dias antes


BOMBA = "data_inicial string, data_final string, produto string, numero_de_postos string, unidade string, " \
        "preco_medio_revenda string"


def test_bomba_semanal_usa_a_segunda_feira_e_o_nome_de_produto_do_projeto(spark):
    bronze = spark.createDataFrame(
        [
            ("2026-11-15", "2026-11-21", "GASOLINA COMUM", "4584", "R$/l", "6.47"),  # 15/11/2026 é domingo
            ("2026-11-15", "2026-11-21", "OLEO DIESEL S10", "3189", "R$/l", "7.26"),
            ("2026-11-15", "2026-11-21", "QUEROSENE", "10", "R$/l", "5.00"),  # produto fora do contrato
            ("2026-11-15", "2026-11-21", "GNV", "358", "R$/m³", "n/d"),  # preço ilegível
            ("2026-11-16", "2026-11-22", "GASOLINA COMUM", "4584", "R$/l", "6.50"),  # semana não começa no domingo
            ("2026-11-08", "2026-11-14", "GASOLINA COMUM", "4504", "R$/l", "NaN"),
            ("2026-11-01", "2026-11-07", "GASOLINA COMUM", "4500", "R$/l", "647"),  # centavos: fora da faixa
        ],
        BOMBA,
    )
    ruins = {(r["data_inicial"], r["produto_texto"]) for r in referencias.bomba_invalidas(bronze).collect()}
    assert ruins == {
        ("2026-11-15", "QUEROSENE"),
        ("2026-11-15", "GNV"),
        ("2026-11-16", "GASOLINA COMUM"),
        ("2026-11-08", "GASOLINA COMUM"),
        ("2026-11-01", "GASOLINA COMUM"),
    }
    semanal = {(r["semana"], r["produto"]): r for r in referencias.bomba_semanal(bronze).collect()}
    segunda = dt.date(2026, 11, 16)  # a mesma semana dos microdados da ANP
    assert set(semanal) == {(segunda, "GASOLINA"), (segunda, "DIESEL S10")}
    assert semanal[(segunda, "GASOLINA")]["preco_medio"] == pytest.approx(6.47)
    assert semanal[(segunda, "DIESEL S10")]["n_postos"] == 3189


def test_tabela_do_repasse_poe_bomba_e_barril_lado_a_lado(spark):
    s1, s2 = dt.date(2026, 11, 9), dt.date(2026, 11, 16)
    bomba = spark.createDataFrame(
        [(s1, "GASOLINA", 6.48, 4504), (s1, "DIESEL S10", 7.29, 3100), (s2, "GASOLINA", 6.47, 4584),
         (s1, "GLP", 110.0, 3000)],  # GLP não entra no repasse
        "semana date, produto string, preco_medio double, n_postos int",
    )
    brent = spark.createDataFrame(
        [(s1, 92.8, 5, 5.59, 5, 3.26), (dt.date(2026, 11, 2), 100.4, 5, 5.48, 5, 3.46)],
        "semana date, brent_usd double, dias_brent long, dolar double, dias_dolar long, brent_reais_litro double",
    )
    tabela = referencias.repasse_semanal(bomba, brent)
    assert set(tabela.columns) == {"semana", "gasolina", "diesel_s10", "diesel", "etanol", "brent_usd", "dias_brent",
                                   "dolar", "dias_dolar", "brent_reais_litro"}
    linhas = {r["semana"]: r for r in tabela.collect()}
    assert linhas[s1]["gasolina"] == pytest.approx(6.48) and linhas[s1]["brent_reais_litro"] == pytest.approx(3.26)
    assert linhas[s2]["gasolina"] == pytest.approx(6.47) and linhas[s2]["brent_reais_litro"] is None
    assert linhas[dt.date(2026, 11, 2)]["gasolina"] is None  # semana só com barril continua na tabela


def test_macro_mensal_agrega_por_mes_e_converte_alimentos_para_reais(spark):
    diario = spark.createDataFrame(
        [(dt.date(2026, 3, 2), 100.0, 5.0), (dt.date(2026, 3, 3), 120.0, 5.2), (dt.date(2026, 4, 1), 90.0, 5.0)], DIARIO
    )
    bomba = spark.createDataFrame(
        [(dt.date(2026, 3, 2), "DIESEL S10", 7.0, 10), (dt.date(2026, 3, 9), "DIESEL S10", 7.4, 10)],
        "semana date, produto string, preco_medio double, n_postos int",
    )
    ipca = referencias.mensal(spark.createDataFrame([("01/03/2026", "1,04"), ("01/04/2026", "n/d")], BRONZE),
                              "ipca_alimentos")
    mundo = referencias.mensal(spark.createDataFrame([("2026-03-01", "130.5")], BRONZE), "alimentos_mundo")
    meses = {
        r["mes"]: r
        for r in referencias.macro_mensal(diario, bomba, {"ipca_alimentos": ipca, "alimentos_mundo": mundo}).collect()
    }
    marco, abril = meses[dt.date(2026, 3, 1)], meses[dt.date(2026, 4, 1)]
    assert marco["brent_usd"] == pytest.approx(110.0) and marco["dolar"] == pytest.approx(5.1)
    assert marco["barril_reais"] == pytest.approx(110.0 * 5.1)
    assert marco["diesel_s10"] == pytest.approx(7.2)
    assert marco["ipca_alimentos"] == pytest.approx(1.04)
    assert marco["alimentos_mundo_reais"] == pytest.approx(130.5 * 5.1)
    assert abril["ipca_alimentos"] is None and abril["diesel_s10"] is None  # valor ilegível não vira número


SERIES_DE_DATA_E_VALOR = [
    ("brent", "%Y-%m-%d", 1000, 7),
    ("dolar", "%d/%m/%Y", 1000, 7),
    ("ipca_geral", "%d/%m/%Y", 100, 31),
    ("ipca_alimentos", "%d/%m/%Y", 100, 31),
    ("alimentos_mundo", "%Y-%m-%d", 100, 31),
]


@pytest.mark.parametrize("fonte,formato,minimo,passo_maximo", SERIES_DE_DATA_E_VALOR)
def test_arquivos_versionados_sao_series_integras(fonte, formato, minimo, passo_maximo):
    """Sem Spark: um download ruim commitado em referencias/ tem de quebrar o CI."""
    config = FONTES_REFERENCIA[fonte]
    with open(RAIZ / "referencias" / ARQUIVOS_REFERENCIA[fonte], encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.reader(f, delimiter=config["sep"]))
    assert linhas[0] == list(config["colunas"])
    corpo = linhas[1:]
    assert len(corpo) > minimo
    assert all(len(linha) == 2 for linha in corpo)

    datas = [dt.datetime.strptime(d, formato).date() for d, _ in corpo]
    assert datas == sorted(set(datas)), "datas repetidas ou fora de ordem"
    assert max(b - a for a, b in zip(datas, datas[1:])) <= dt.timedelta(days=passo_maximo), "buraco na série"
    if passo_maximo == 7:
        assert all(d.weekday() < 5 for d in datas)
    else:
        assert all(d.day == 1 for d in datas)

    faixa_minima, faixa_maxima = config["faixa"]
    valores = [float(v.replace(",", ".")) for _, v in corpo if v not in ("", ".")]
    assert len(valores) > 0.95 * len(corpo)
    assert all(faixa_minima <= v <= faixa_maxima for v in valores)


def test_arquivo_versionado_da_bomba_e_integro():
    with open(RAIZ / "referencias" / ARQUIVOS_REFERENCIA["bomba_brasil"], encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    assert list(linhas[0]) == COLUNAS_BOMBA
    assert len(linhas) > 4000
    inicios = [dt.date.fromisoformat(linha["data_inicial"]) for linha in linhas]
    assert all(d.weekday() == 6 for d in inicios), "a semana da ANP começa no domingo"
    assert {linha["produto"] for linha in linhas} <= set(PRODUTOS_BOMBA)
    minimo, maximo = FAIXA_PRECO_BOMBA
    assert all(minimo <= float(linha["preco_medio_revenda"]) <= maximo for linha in linhas)
    gasolina = sorted({d for d, linha in zip(inicios, linhas) if linha["produto"] == "GASOLINA COMUM"})
    buracos = [(a, b) for a, b in zip(gasolina, gasolina[1:]) if (b - a).days != 7]
    # as duas interrupções documentadas pela ANP: 1 semana em ago/2015 e a pesquisa suspensa em 2020
    assert buracos == [(dt.date(2015, 8, 9), dt.date(2015, 8, 23)), (dt.date(2020, 8, 16), dt.date(2020, 10, 18))]
