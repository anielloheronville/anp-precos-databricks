"""Onde as tabelas moram.

- Databricks: tabelas Delta no Unity Catalog (`catalogo` informado).
- Local: Parquet numa pasta (`pasta_local`), para Linux/macOS ou CI.
- Memória (`em_memoria=True`): checkpoint local em memória. Serve para rodar no
  Windows sem o winutils.exe do Hadoop, que o Spark exige para gravar em disco.

O mesmo código de transformação roda nos três modos; só a gravação muda.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


class Lago:
    def __init__(
        self,
        spark: SparkSession,
        *,
        catalogo: str | None = None,
        esquema: str = "anp",
        pasta_local: str | Path | None = None,
        em_memoria: bool = False,
    ) -> None:
        modos = sum([catalogo is not None, pasta_local is not None, em_memoria])
        if modos != 1:
            raise ValueError("Escolha um modo: catalogo (Databricks), pasta_local ou em_memoria.")
        self.spark = spark
        self.catalogo = catalogo
        self.esquema = esquema
        self.pasta_local = Path(pasta_local) if pasta_local else None
        self._memoria: dict[str, DataFrame] | None = {} if em_memoria else None

    @property
    def no_databricks(self) -> bool:
        return self.catalogo is not None

    def nome(self, tabela: str) -> str:
        if self.no_databricks:
            return f"{self.catalogo}.{self.esquema}.{tabela}"
        if self._memoria is not None:
            return f"memoria:{tabela}"
        return str(self.pasta_local / tabela)

    def gravar(
        self,
        df: DataFrame,
        tabela: str,
        *,
        comentario: str | None = None,
        cluster_por: list[str] | None = None,
    ) -> str:
        destino = self.nome(tabela)
        if self.no_databricks:
            escrita = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
            if cluster_por and hasattr(escrita, "clusterBy"):
                escrita = escrita.clusterBy(*cluster_por)  # liquid clustering do Delta
            escrita.saveAsTable(destino)
            if comentario:
                texto = comentario.replace("'", "\\'")
                self.spark.sql(f"COMMENT ON TABLE {destino} IS '{texto}'")
        elif self._memoria is not None:
            # materializa agora e corta a linhagem, como uma tabela gravada faria
            self._memoria[tabela] = df.localCheckpoint(eager=True)
        else:
            df.write.mode("overwrite").parquet(destino)
        return destino

    def ler(self, tabela: str) -> DataFrame:
        if self.no_databricks:
            return self.spark.table(self.nome(tabela))
        if self._memoria is not None:
            if tabela not in self._memoria:
                raise KeyError(f"Tabela {tabela} ainda não foi gravada nesta sessão.")
            return self._memoria[tabela]
        return self.spark.read.parquet(self.nome(tabela))
