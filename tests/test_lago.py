import os

import pytest

from anp_precos.lago import Lago


def test_exige_exatamente_um_modo(spark, tmp_path):
    with pytest.raises(ValueError):
        Lago(spark)
    with pytest.raises(ValueError):
        Lago(spark, catalogo="workspace", pasta_local=tmp_path)


def test_memoria_grava_e_le(spark):
    lago = Lago(spark, em_memoria=True)
    lago.gravar(spark.range(5), "t")
    assert lago.ler("t").count() == 5
    with pytest.raises(KeyError):
        lago.ler("nao_existe")


@pytest.mark.skipif(os.name == "nt", reason="Spark no Windows precisa do winutils.exe para gravar em disco")
def test_parquet_grava_e_le(spark, tmp_path):
    lago = Lago(spark, pasta_local=tmp_path)
    lago.gravar(spark.range(3), "t")
    assert sorted(r["id"] for r in lago.ler("t").collect()) == [0, 1, 2]


def test_nome_no_databricks(spark):
    assert Lago(spark, catalogo="workspace").nome("silver_precos") == "workspace.anp.silver_precos"
