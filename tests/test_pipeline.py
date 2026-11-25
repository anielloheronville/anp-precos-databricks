"""O pipeline de ponta a ponta em memória: referências, gold e a ligação entre os dois."""

import datetime as dt

import pytest

from anp_precos import pipeline
from anp_precos.lago import Lago

SILVER = (
    "regiao string, uf string, municipio string, cnpj string, produto string, "
    "semana date, valor_venda double, bandeira string"
)
S0 = dt.date(2026, 3, 2)  # segunda-feira


def _semana(i: int) -> dt.date:
    return S0 + dt.timedelta(days=7 * i)


def _referencias(tmp_path, semanas: dict[int, float], dolar: float = 5.0) -> tuple[str, str]:
    """Arquivos nos formatos reais, com 5 cotações por semana. `semanas`: índice -> Brent em US$."""
    brent, cambio = ["observation_date,DCOILBRENTEU"], ['"data";"valor"']
    for i, preco in sorted(semanas.items()):
        for dia in range(5):
            data = _semana(i) + dt.timedelta(days=dia)
            brent.append(f"{data:%Y-%m-%d},{preco:.2f}")
            cambio.append(f'"{data:%d/%m/%Y}";"{dolar:.4f}"'.replace(".", ","))
    arq_brent, arq_dolar = tmp_path / "brent.csv", tmp_path / "dolar.csv"
    arq_brent.write_bytes(("\n".join(brent) + "\n").encode("utf-8"))
    arq_dolar.write_bytes(("\r\n".join(cambio) + "\r\n").encode("utf-8"))
    return str(arq_brent), str(arq_dolar)


def _lago_com_silver(spark, semanas: list[int]) -> Lago:
    lago = Lago(spark, em_memoria=True)
    linhas = [
        ("CO", "MT", "SORRISO", str(posto), "GASOLINA", _semana(i), 6.0 + 0.1 * i, "VIBRA")
        for i in semanas
        for posto in range(3)
    ]
    lago.gravar(spark.createDataFrame(linhas, SILVER), "silver_precos")
    return lago


def test_referencias_viram_tabelas_e_resumo(spark, tmp_path):
    lago = Lago(spark, em_memoria=True)
    resumo = pipeline.rodar_referencias(spark, lago, *_referencias(tmp_path, {0: 80.0, 1: 88.0, 2: 96.8}))

    assert resumo == {
        "dias_brent": 15,
        "dias_dolar": 15,
        "semanas": 3,
        "semanas_com_preco": 3,
        "primeira_semana_com_preco": str(_semana(0)),
        "ultima_semana_com_preco": str(_semana(2)),
    }
    semanal = {r["semana"]: r for r in lago.ler("gold_brent_semanal").collect()}
    assert semanal[_semana(2)]["var_brent_1s"] == pytest.approx(0.10)
    assert lago.ler("bronze_brent").count() == 15  # o bronze guarda uma linha por cotação, como veio


def test_linha_invalida_derruba_a_carga_das_referencias(spark, tmp_path):
    arq_brent, arq_dolar = _referencias(tmp_path, {0: 80.0, 1: 88.0})
    with open(arq_brent, "a", encoding="utf-8") as f:
        f.write(f"{_semana(2):%Y-%m-%d},n/d\n")
    with pytest.raises(ValueError, match="Série 'brent' tem linhas"):
        pipeline.rodar_referencias(spark, Lago(spark, em_memoria=True), arq_brent, arq_dolar)


def test_arquivo_so_com_cabecalho_derruba_a_carga(spark, tmp_path):
    arq_brent, arq_dolar = _referencias(tmp_path, {0: 80.0, 1: 88.0})
    with open(arq_brent, "w", encoding="utf-8") as f:
        f.write("observation_date,DCOILBRENTEU\n")
    with pytest.raises(ValueError, match="Nenhuma semana com barril em reais"):
        pipeline.rodar_referencias(spark, Lago(spark, em_memoria=True), arq_brent, arq_dolar)


def test_gold_sem_referencias_falha_antes_de_gravar(spark):
    lago = _lago_com_silver(spark, [3, 4])
    with pytest.raises(RuntimeError, match="rode pipeline.rodar_referencias antes"):
        pipeline.rodar_gold(lago)
    with pytest.raises(KeyError):  # nada do gold foi gravado
        lago.ler("gold_mercado_semanal")


def test_gold_liga_o_barril_as_features_com_uma_semana_de_atraso(spark, tmp_path):
    lago = _lago_com_silver(spark, [3, 4, 5])
    # Brent nas semanas 0 a 5: sobe devagar até a 4 e dá um salto de +50% na semana 5
    precos = {0: 80.0, 1: 88.0, 2: 96.8, 3: 106.48, 4: 117.13, 5: 175.70}
    pipeline.rodar_referencias(spark, lago, *_referencias(tmp_path, precos))
    cobertura = pipeline.rodar_gold(lago)

    assert cobertura == {"semanas": 3, "semanas_sem_brent": 0}
    linhas = {r["semana"]: r for r in lago.ler("gold_features").collect()}
    assert len(linhas) == 3
    # a linha da semana 5 usa o barril até a semana 4; o salto de +50% só apareceria na semana 6
    assert linhas[_semana(5)]["var_brent_1s"] == pytest.approx(117.13 / 106.48 - 1)
    assert linhas[_semana(4)]["var_brent_1s"] == pytest.approx(106.48 / 96.8 - 1)
    assert linhas[_semana(5)]["var_brent_4s"] == pytest.approx(117.13 / 80.0 - 1)
    assert linhas[_semana(3)]["var_brent_4s"] is None  # precisaria da semana -2
    assert linhas[_semana(3)]["sobe_prox_semana"] == 1  # 6,30 -> 6,40: o alvo continua o mesmo


def _series_macro(tmp_path, semanas: list[int], preco_invalido: bool = False) -> dict[str, str]:
    """Arquivos das séries mensais e da bomba semanal, nos formatos reais."""
    bcb = ['"data";"valor"', '"01/03/2026";"1,04"', '"01/04/2026";"0,52"']
    fmi = ["observation_date,PFOODINDEXM", "2026-03-01,130.5", "2026-04-01,131.0"]
    bomba = ["data_inicial,data_final,produto,numero_de_postos,unidade,preco_medio_revenda"]
    for i in semanas:
        domingo = _semana(i) - dt.timedelta(days=1)
        sabado = domingo + dt.timedelta(days=6)
        bomba.append(f"{domingo},{sabado},GASOLINA COMUM,4500,R$/l,{6.0 + 0.1 * i:.2f}")
        bomba.append(f"{domingo},{sabado},OLEO DIESEL S10,3100,R$/l,{'abc' if preco_invalido else '7.00'}")
    conteudos = {"ipca_geral": bcb, "ipca_alimentos": bcb, "alimentos_mundo": fmi, "bomba_brasil": bomba}
    caminhos = {}
    for serie, linhas in conteudos.items():
        arquivo = tmp_path / f"{serie}.csv"
        arquivo.write_bytes(("\n".join(linhas) + "\n").encode("utf-8"))
        caminhos[serie] = str(arquivo)
    return caminhos


def test_macro_cria_as_tabelas_do_repasse_e_dos_eventos(spark, tmp_path):
    lago = Lago(spark, em_memoria=True)
    pipeline.rodar_referencias(spark, lago, *_referencias(tmp_path, {0: 80.0, 1: 88.0, 2: 96.8, 4: 100.0}))
    resumo = pipeline.rodar_macro(spark, lago, _series_macro(tmp_path, [1, 2, 3]))

    assert resumo == {
        "semanas_com_bomba": 3,
        "semanas_com_bomba_e_barril": 2,  # a semana 3 tem bomba, mas não tem barril
        "ultima_semana_da_bomba": str(_semana(3)),
        "meses_com_ipca_de_alimentos": 2,
        "ultimo_mes_do_ipca": "2026-04",
    }
    semanal = {r["semana"]: r for r in lago.ler("gold_repasse_semanal").collect()}
    assert semanal[_semana(1)]["gasolina"] == pytest.approx(6.10)  # domingo da ANP -> segunda do projeto
    assert semanal[_semana(1)]["brent_usd"] == pytest.approx(88.0)
    assert semanal[_semana(3)]["brent_reais_litro"] is None and semanal[_semana(0)]["gasolina"] is None
    mensal = {r["mes"]: r for r in lago.ler("gold_macro_mensal").collect()}
    assert mensal[dt.date(2026, 3, 1)]["ipca_alimentos"] == pytest.approx(1.04)
    assert mensal[dt.date(2026, 3, 1)]["alimentos_mundo_reais"] == pytest.approx(130.5 * 5.0)


def test_macro_sem_referencias_e_com_linha_invalida_falham_com_mensagem(spark, tmp_path):
    with pytest.raises(RuntimeError, match="rode pipeline.rodar_referencias antes"):
        pipeline.rodar_macro(spark, Lago(spark, em_memoria=True), _series_macro(tmp_path, [1]))

    lago = Lago(spark, em_memoria=True)
    pipeline.rodar_referencias(spark, lago, *_referencias(tmp_path, {0: 80.0, 1: 88.0}))
    with pytest.raises(ValueError, match="Série 'bomba_brasil' tem linhas"):
        pipeline.rodar_macro(spark, lago, _series_macro(tmp_path, [1], preco_invalido=True))

    # série mensal só com cabeçalho também não passa em silêncio
    caminhos = _series_macro(tmp_path, [1])
    with open(caminhos["alimentos_mundo"], "w", encoding="utf-8") as f:
        f.write("observation_date,PFOODINDEXM\n")
    with pytest.raises(ValueError, match="Série 'alimentos_mundo' sem nenhum mês válido"):
        pipeline.rodar_macro(spark, lago, caminhos)


def test_brent_desatualizado_derruba_o_gold(spark, tmp_path):
    lago = _lago_com_silver(spark, [10, 11, 12])
    pipeline.rodar_referencias(spark, lago, *_referencias(tmp_path, {0: 80.0, 1: 88.0, 2: 96.8}))
    with pytest.raises(ValueError, match="falta em 3 de 3 semanas"):
        pipeline.rodar_gold(lago)
