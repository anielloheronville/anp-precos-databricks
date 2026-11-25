import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Sem isto o PySpark chama "python3" para os workers. No Windows esse nome é o atalho da
# Microsoft Store, e todo teste que usa createDataFrame falha com "Python worker failed to connect back".
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    sessao = (
        SparkSession.builder.master("local[2]")
        .appName("testes-anp")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    sessao.sparkContext.setLogLevel("ERROR")
    yield sessao
    sessao.stop()
