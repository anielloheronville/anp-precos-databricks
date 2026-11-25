# Radar de Preços de Combustíveis · PySpark + Databricks

> Estudo independente sobre dados públicos da ANP, sem relação com nenhuma distribuidora.
> Bandeira conforme declarada à ANP; as marcas pertencem aos seus titulares.

Pipeline de dados e modelo preditivo sobre a **pesquisa semanal de preços da ANP**: 3 milhões
de coletas de jan/2023 a jun/2026, posto a posto, com a bandeira de cada revenda.

Perguntas de negócio:

1. **Onde a Ipiranga está acima ou abaixo do mercado?** Em cada município e semana, a mediana
   dos postos Ipiranga contra a mediana dos concorrentes do mesmo município.
2. **O preço do município vai subir na semana que vem?** Classificação para antecipar altas de
   pelo menos 0,5% na mediana municipal.
3. **O barril de petróleo ajuda a antecipar essa alta?** O Brent em reais entra como sinal, e o
   modelo é medido com e sem ele.
4. **Quanto do barril chega à bomba, e o que isso implica se ele subir ou cair?** Repasse medido
   semana a semana e cenários condicionais para gasolina e diesel.
5. **A alta dos alimentos depois da pandemia acompanha a COVID e as guerras?** Estudo de eventos com
   dólar, barril, diesel e alimentos.

## Resultados

### Qualidade dos dados

| Etapa | Linhas |
|---|---:|
| Bronze (como veio da ANP) | 3.038.685 |
| Quarentena (linhas `;;;;` vazias no semestre 2025-01) | 9.114 |
| Duplicatas removidas (mesmo posto, produto e dia) | 20 |
| **Silver (válidas)** | **3.029.551** |

Três armadilhas da fonte que o pipeline trata e os testes cobrem: a ANP trocou nomes de bandeira
no meio da série (`VIBRA ENERGIA` → `VIBRA`, `ALESAT` → `ALE`), o semestre 2025-02 grava a unidade
do GNV como `R$ / m3` em vez de `R$ / m³`, e há campos entre aspas com `;` e quebra de linha.

### Prêmio da Ipiranga sobre os concorrentes do mesmo município

| Produto | Municípios | Prêmio no município típico | IC 95% (bootstrap por município) | Acima / igual / abaixo |
|---|---:|---:|---|---|
| Diesel S10 | 405 | +1,0% | +0,7% a +1,6% | 61% / 10% / 29% |
| Diesel | 362 | +1,1% | +0,8% a +1,5% | 61% / 7% / 32% |
| Gasolina aditivada | 392 | +1,0% | +0,8% a +1,4% | 64% / 8% / 27% |
| Etanol | 388 | +0,9% | +0,4% a +1,6% | 61% / 16% / 23% |
| Gasolina comum | 408 | +0,3% | +0,2% a +0,6% | 60% / 18% / 22% |
| GNV | 55 | 0,0% | 0,0% a 0,0% | 38% / 31% / 32% |

O prêmio é pequeno na gasolina comum, o produto mais comparado pelo consumidor, e maior em
diesel e aditivada. Varia muito por estado (em SP passa de 2,5% na gasolina; ES e AC ficam abaixo
do mercado): ver `reports/figuras/premio_uf_*.png`.

![Prêmio por UF](reports/figuras/premio_uf_gasolina.png)

### Modelo: alta de ≥ 0,5% na semana seguinte

Teste nas últimas 26 semanas (dez/2025 a jun/2026), nunca usado para escolha alguma. Taxa de alta
no teste: 24%.

| Modelo | ROC AUC | Average precision | Precisão nos 10% mais prováveis |
|---|---:|---:|---:|
| Gradient boosting com o barril em reais | **0,724** | **0,494** | **62%** |
| Gradient boosting sem o barril | 0,690 | 0,382 | 49% |
| Regressão logística sem o barril | 0,683 | 0,437 | 53% |
| Regressão logística com o barril | 0,652 | 0,387 | 46% |
| Persistência (variação da última semana) | 0,518 | 0,300 | 42% |

O "barril em reais" são três features: a variação do Brent × dólar em 1, 4 e 8 semanas, medida até
a semana anterior. Os dois modelos foram treinados de novo sem elas, com a mesma partição, a mesma
semente e os mesmos hiperparâmetros, para medir a diferença.

Leitura honesta:

- **O barril ajuda o boosting.** O AUC sobe de 0,739 para 0,752 na validação e de 0,690 para 0,724
  no teste. Reamostrando semanas inteiras, o ganho é de +0,013 na validação (intervalo de 90%: +0,004
  a +0,024) e de +0,034 no teste (+0,004 a +0,068). No Databricks a mesma comparação deu +0,015 na
  validação (+0,007 a +0,024) e +0,028 no teste (-0,002 a +0,063): o ganho da validação se repete,
  o do teste é positivo mas o intervalo encosta no zero. A decisão de manter a feature se apoia na
  validação. Entre os 10% de mercados mais prováveis, 62% subiram de fato, contra 24% de média.
- **O ganho do teste depende de um episódio.** O teste contém a guerra do Irã: o Brent foi de US$ 70
  a mais de US$ 120 entre fevereiro e março de 2026. Em 12 das 26 semanas do teste a variação de 8
  semanas do barril está fora de tudo o que o treino viu (até +76%, contra +21% no treino), e ali
  uma árvore só repete o que aprendeu na ponta. O ganho da validação, em período calmo, é a
  estimativa conservadora.
- **A logística piora com o barril** (0,683 para 0,652 no teste, e também na validação). O efeito
  não é linear, e por isso a referência linear da tabela é a logística sem o barril. O boosting, que
  antes empatava com ela, agora ganha com folga.
- **O barril diz quando, não onde.** Ele é igual para todos os municípios da semana: aponta as
  semanas e os produtos em que a alta é mais provável, não distingue um município do vizinho. Por
  produto, o AUC no teste com e sem o barril: gasolina aditivada 0,698 e 0,640; gasolina comum 0,698
  e 0,654; diesel S10 0,733 e 0,697; etanol 0,715 e 0,692; diesel 0,759 e 0,738; GNV 0,678 e 0,677.
- **O repasse leva semanas.** A variação nacional da última semana continua sendo o que mais pesa;
  depois vêm a variação local, a dispersão de preços no município e o barril em 4 e 8 semanas. A
  variação do barril em 1 semana quase não pesa. A semana do ano continua atrapalhando: 2,5 anos de
  treino não bastam para aprender sazonalidade.
- **Limites da medida.** Os horizontes de 1, 4 e 8 semanas foram fixados antes de olhar qualquer
  métrica do modelo, mas depois de uma análise exploratória do repasse que usou a série inteira,
  inclusive o período de teste. E a ordem das linhas que sai do Spark muda o boosting na terceira
  casa (cerca de 0,001 de AUC entre execuções idênticas).

![Importância por permutação](reports/figuras/importancia_features.png)

### O barril chega à bomba?

Preço médio semanal de revenda no Brasil (ANP) contra o barril de Brent em reais por litro, de 2017
a novembro de 2026. Em 2026 o barril em reais quase dobrou entre o piso e o pico; a gasolina variou
8% e o diesel S10, 24%.

![Barril e bomba](reports/figuras/barril_e_bomba.png)

Repasse acumulado para cada 1% de alta do barril em reais (entre parênteses, o intervalo de 90%):

| Produto | Período | 4 semanas | 8 semanas | 16 semanas |
|---|---|---:|---:|---:|
| Gasolina comum | até mai/2023 (paridade de importação) | 0,17% | 0,23% | 0,30% (0,20 a 0,44) |
| Gasolina comum | de mai/2023 em diante | 0,09% | 0,07% | 0,06% (-0,13 a 0,18) |
| Diesel S10 | até mai/2023 (paridade de importação) | 0,16% | 0,22% | 0,33% (0,21 a 0,48) |
| Diesel S10 | de mai/2023 em diante | 0,25% | 0,29% | 0,35% (-0,02 a 0,62) |

- Até maio de 2023, gasolina e diesel repassavam cerca de 30% de uma alta do barril em quatro meses.
- Depois que a Petrobras deixou a paridade de importação, o repasse da gasolina caiu para 6% e não
  se distingue de zero. O do diesel continuou perto de 35% e ficou mais rápido, mas com intervalo
  largo: são 183 semanas e poucos choques grandes.
- A conta é uma regressão das variações semanais da bomba nas variações do barril da mesma semana e
  das 16 anteriores. Ela mede o que aconteceu em média no período e não separa mudanças de imposto
  (os cortes de 2022, a volta dos tributos em 2023) nem decisões de preço.

![Repasse do barril](reports/figuras/repasse_do_barril.png)

### Cenários: e se o barril subir ou cair?

Ponto de partida: semana de 09/11/2026, com o Brent a US$ 92,8, o dólar a R$ 5,59, a gasolina a
R$ 6,48 e o diesel S10 a R$ 7,29. Preço na bomba 16 semanas depois do salto do barril (intervalo de
90%):

| Cenário | Barril em reais | Gasolina, repasse de hoje | Diesel S10, repasse de hoje | Gasolina, se a paridade voltasse |
|---|---:|---|---|---|
| Barril e dólar ficam onde estão | 0% | R$ 6,41 (6,33 a 6,47) | R$ 7,05 (6,82 a 7,20) | R$ 6,25 (6,13 a 6,32) |
| Escalada: Brent volta ao pico de 12 meses (US$ 127) | +37% | R$ 6,53 (6,17 a 6,76) | R$ 7,88 (7,01 a 8,55) | R$ 6,87 (6,64 a 7,14) |
| Escalada com dólar no máximo da série (R$ 6,19) | +52% | R$ 6,57 (6,09 a 6,89) | R$ 8,17 (6,99 a 9,10) | R$ 7,09 (6,79 a 7,47) |
| Alívio: Brent volta ao piso de 12 meses (US$ 61) | -34% | R$ 6,25 (5,88 a 6,80) | R$ 6,09 (5,36 a 7,18) | R$ 5,51 (5,12 a 5,75) |

- **O risco está no diesel.** Numa nova escalada, ele sobe cerca de 8% (R$ 0,59 por litro) e a
  gasolina, menos de 1%.
- **A política de preços pesa tanto quanto o barril.** A mesma escalada dá +0,8% na gasolina com o
  repasse de hoje e +6% se a paridade de importação voltasse.
- **"Fica onde está" ainda cai um pouco** (1% na gasolina, 3% no diesel). O barril recuou de cerca
  de US$ 124 na média de outubro para US$ 93, e parte desse recuo ainda não chegou à bomba.
- **O ponto de partida está um pouco atrás do câmbio.** A última semana com Brent completo na fonte
  é a de 09/11. O dólar médio da semana de 23/11 já era R$ 5,90, cerca de 6% acima.
- **São contas condicionais, não previsões.** As hipóteses saem das semanas completas da própria
  série (pico e piso do Brent em 12 meses, maior média semanal do dólar), o barril salta de uma vez
  e fica lá, o repasse é o mesmo para alta e para queda, e imposto, tendência do preço e mudança de
  política ficam de fora.

![Cenários](reports/figuras/cenarios_bomba.png)

### COVID, guerras e alimentos

O que mudou nos seis meses seguintes ao mês anterior a cada choque:

| Evento | Dólar | Barril em reais | Diesel S10 | Alimentos no mundo (R$) | Alimentos no Brasil | Inflação geral |
|---|---:|---:|---:|---:|---:|---:|
| COVID (mar/2020) | +26% | +1% | -10% | +23% | +4,4% | +0,2% |
| Ucrânia (fev/2022) | -3% | +26% | +36% | -1% | +8,6% | +4,2% |
| Gaza (out/2023) | +1% | -8% | -4% | +4% | +5,0% | +2,5% |
| Irã, 12 dias (jun/2025) | -6% | -7% | -1% | -9% | -1,2% | +1,1% |
| Irã, Ormuz (mar/2026) | -1% | +27% | +13% | +2% | +3,0% | +2,1% |

- **COVID:** em 12 meses os alimentos subiram 15,0% no Brasil, contra 5,2% da inflação geral. O
  diesel estava caindo: a alta veio com o dólar (+26% em seis meses) e com os alimentos no mundo
  medidos em reais (+48% em 12 meses), não pelo combustível.
- **Ucrânia:** o diesel subiu 36% em seis meses e os alimentos, 8,6%, o dobro da inflação geral.
- **Gaza e Irã em 2025:** o barril em reais caiu depois dos dois eventos. Não há canal de
  combustível visível.
- **Irã e Ormuz em 2026:** barril +27%, diesel +13% e alimentos +3,0% em seis meses, contra 1,8% nos
  12 meses anteriores.
- **Correlação existe, mas é modesta.** A inflação de alimentos de um mês acompanha a variação do
  diesel de um e dois meses antes (0,24 e 0,23) e a dos alimentos no mundo do próprio mês e do
  anterior (0,27 e 0,32 em dólares; 0,17 e 0,28 em reais). Barril e gasolina ficam abaixo de 0,10,
  e com cerca de 115 meses nada abaixo de 0,18 se distingue de zero. Uma regressão em dois canais
  (alimentos no mundo em reais e diesel) explica 23% da variação mensal, ou 15% descontado o número
  de regressores.
- **O que sobra é grande.** Entre o período anterior à COVID e março de 2020 a dezembro de 2022, a
  inflação de alimentos foi de 0,23% para 0,93% ao mês. A contribuição dos dois canais passou de
  0,16 para 0,38 ponto: cerca de um terço da aceleração. O resto é o que o modelo não mede (safra,
  clima, demanda).
- **Um mês interpolado.** A ANP não pesquisou preços em setembro de 2020. Sem tratamento, a falta
  desse único mês de diesel tiraria da regressão seis meses, justamente os de maior inflação de
  alimentos. O diesel daquele mês é interpolado entre agosto e outubro; as médias por período usam
  todos os meses.
- **É descritivo.** Um evento é uma data, e os choques se sobrepõem. A tabela mostra o que aconteceu
  depois de cada um, não prova a causa.

![Eventos e alimentos](reports/figuras/eventos_e_alimentos.png)

## Arquitetura (medallion no Unity Catalog)

```
ANP (zip) ─► Volume UC  /Volumes/workspace/anp/bruto  (descompactado no próprio volume)
                     │
                     ▼
   bronze_precos        texto como veio + arquivo de origem
                     │   regras de qualidade, try_cast (ANSI), deduplicação
                     ▼
   silver_precos        tipado, 1 linha por posto/produto/dia   ─┐
   silver_quarentena    linhas rejeitadas + motivo               │ liquid clustering
                     │                                           │ (produto, semana)
                     ▼                                           │
   gold_mercado_semanal   UF × município × produto × semana     ─┘
   gold_bandeira_semanal  mediana nacional por bandeira
   gold_nacional_semanal  mediana nacional e variações
   gold_features          features + alvo para o modelo  ◄── gold_brent_semanal
                     │
                     ▼
   scikit-learn + SciPy + MLflow   (modelo, métricas, gráficos)

Brent (EIA) e dólar (Banco Central) ─► referencias/*.csv no repositório ─► Volume UC
   bronze_brent, bronze_dolar      texto como veio de cada fonte
   silver_referencias_diarias      1 linha por dia: brent_usd, dolar
   gold_brent_semanal              barril em reais por litro e variações de 1, 4 e 8 semanas

Bomba semanal (ANP), IPCA (IBGE) e alimentos no mundo (FMI) ─► referencias/*.csv ─► Volume UC
   bronze_bomba_brasil, bronze_ipca_geral, bronze_ipca_alimentos, bronze_alimentos_mundo
   silver_bomba_brasil             preço médio de revenda no Brasil por semana e produto
   gold_repasse_semanal            bomba × barril em reais, por semana  ─► repasse e cenários
   gold_macro_mensal               dólar, barril, bomba, IPCA e alimentos, por mês ─► estudo de eventos
```

O mesmo pacote `src/anp_precos` roda **localmente** e **no Databricks** (Delta no Unity Catalog).
Só a classe `Lago` sabe onde gravar. O CI roda as transformações e o modelo em Linux (testes de
bronze, silver, gold, séries de referência, pipeline de ponta a ponta em memória, gravação em Parquet
e modelo); a execução completa sobre os 3 milhões de linhas foi feita localmente (cerca de 5 minutos
em 8 núcleos, já com o modelo treinado duas vezes para a comparação com e sem o barril) e no
Databricks serverless (notebooks 01 e 02 em cerca de 4 minutos, e o 03 em cerca de 3). As contagens
de qualidade, as séries de referência, a regressão logística, o repasse e os cenários saem idênticos
nos dois ambientes; o gradient boosting difere na terceira casa.

## Decisões que importam

- **Leitura defensiva:** BOM, `;` como separador, vírgula decimal, campos entre aspas com quebra de
  linha e cabeçalho conferido em cada arquivo (`enforceSchema=false`).
- **ANSI ligado:** Spark 4 e o serverless do Databricks lançam erro em cast inválido. Todo parse
  usa `try_cast`/`try_to_timestamp`, e a linha ruim vai para a quarentena com o motivo, em vez de
  derrubar o job ou virar nulo silencioso.
- **Mediana exata:** `percentile_approx(x, 0.5)` devolve a mediana inferior. Com 2 postos Ipiranga
  a R$ 6,00 e R$ 6,20, ela daria 6,00 e enviesaria o prêmio. Os grupos município × semana são
  pequenos, então o pipeline usa `median`, exata.
- **Semana faltante não é semana anterior:** a ANP não visita todo município toda semana. As
  variações só são calculadas quando a semana anterior existe de fato.
- **Prêmio justo:** a Ipiranga só é comparada com concorrentes do mesmo município e semana, e só
  quando há pelo menos 2 concorrentes. `RAIZEN MIME` conta como Raízen.
- **Validação no tempo:** treino até jun/2025, validação (que também decide o early stopping) até
  dez/2025, teste até jun/2026, com 1 semana de embargo em cada fronteira, porque o alvo da semana
  t usa o preço de t+1. Nada é embaralhado.
- **Barril em reais só com o que a fonte já publicou:** a EIA divulga o Brent spot com cerca de uma
  semana de atraso. Por isso a linha da semana t recebe a variação do barril medida até t-1, que é
  o que o próprio pipeline conseguiria produzir na hora de prever t+1. Com uma cotação em tempo real
  daria para usar a semana t; aqui a escolha foi não medir o modelo com um dado que ele não teria.
- **Semana incompleta não vira média:** a semana só tem barril em reais com pelo menos 3 cotações
  do Brent e 3 do dólar, e a variação só existe contra a semana exata (7, 28 ou 56 dias antes).
- **Série de referência ruim derruba a carga:** linha com coluna a mais ou a menos, texto no lugar de
  número, NaN, valor fora da faixa plausível ou arquivo vazio param o pipeline com a mensagem. Se o
  barril faltar em mais de 10% das semanas do `gold_features`, o gold também falha, em vez de gravar
  a feature nula em silêncio.
- **A mesma semana nas duas fontes da ANP:** a série semanal data a semana pelo domingo, e os
  microdados têm coletas de segunda a sexta. A série entra como domingo mais um dia, a mesma
  segunda-feira dos microdados.
- **Cenário não é previsão:** as hipóteses de Brent e dólar saem da própria série e mudam sozinhas
  quando os dados são atualizados. Cada preço sai com intervalo de 90%, e o repasse entra em dois
  regimes porque a política de preços muda o resultado mais do que o barril. Se o preço na bomba
  estiver mais de duas semanas atrás do barril, o cenário não é calculado.
- **Planilha sem dependência nova:** a série semanal da ANP só existe em `.xlsx`. O download converte
  para CSV com um leitor de 100 linhas sobre a biblioteca padrão, testado com uma planilha montada
  no próprio teste, em vez de exigir um pacote de Excel em três ambientes.
- **Estatística sem exagero:** semanas de um mesmo município não são independentes, então o
  intervalo de confiança vem de bootstrap reamostrando municípios inteiros, não do p-valor.
- **Revisão adversarial:** antes de publicar, uma revisão independente procurou quebrar o projeto
  e achou a mediana inferior, os nomes antigos de bandeira, o GNV `m3` e o early stopping com
  amostra aleatória. Todos foram corrigidos e ganharam teste. Na entrada do barril em reais, uma
  segunda revisão achou que a feature usava uma semana que a fonte ainda não teria publicado, que
  `NaN` passava pela validação das séries e que três testes passariam com o código errado. Na parte
  de repasse e cenários, uma terceira achou uma média subestimada no texto (seis meses de 2020 fora
  da conta), um comando de atualização que esvaziaria o estudo de eventos sem dar erro, "16 semanas"
  com dois significados e 14 de 16 mutações do código que os testes não pegavam. Tudo corrigido
  antes de publicar.

## Como rodar

### Databricks (Free Edition)

1. *Workspace → Create → Git folder* com a URL deste repositório.
2. Rode `notebooks/00_preparar_unity_catalog`. Suba os zips da ANP (`ca-AAAA-SS.zip`) em
   *Catalog → workspace → anp → Volumes → bruto* e rode a última célula do notebook 00. O Brent e o
   dólar não precisam de upload: o notebook copia `referencias/*.csv` do repositório para o volume.
3. Rode `01_bronze_silver_gold` e depois `02_analise_e_modelo`.
4. Rode `03_repasse_e_cenarios`. Ele só usa as séries de `referencias/` e só depende do notebook 00.

### Local

```bash
python -m venv .venv && .venv/Scripts/activate      # Java 17 instalado
pip install -r requirements.txt
pytest
python scripts/rodar_local.py --origem data/raw/csv  # CSVs da ANP descompactados nessa pasta
python scripts/rodar_local.py --etapas repasse       # só repasse, cenários e eventos (1 a 2 minutos)
python scripts/baixar_referencias.py                 # opcional: atualiza as séries em referencias/
```

No Windows o script usa `--modo memoria` (checkpoint em memória), porque o Spark só grava em disco
no Windows com o `winutils.exe` do Hadoop. Em Linux e macOS o padrão é `--modo parquet`.

## Fonte

- [ANP · Série histórica de preços de combustíveis](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis),
  dados abertos. Esses arquivos não estão no repositório (`data/` no `.gitignore`).
- Brent spot (Europe Brent FOB, US$/barril): U.S. Energy Information Administration, pela série
  [DCOILBRENTEU do FRED](https://fred.stlouisfed.org/series/DCOILBRENTEU).
- Dólar PTAX de venda: Banco Central do Brasil, [série 1 do SGS](https://www3.bcb.gov.br/sgspub/).
- Preço médio semanal de revenda no Brasil: ANP,
  [série histórica do levantamento de preços](https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/precos-revenda-e-de-distribuicao-combustiveis/serie-historica-do-levantamento-de-precos).
- IPCA geral e de alimentação e bebidas: IBGE, pelas séries 433 e 1635 do SGS do Banco Central.
- Índice mundial de preços de alimentos: FMI, pela série
  [PFOODINDEXM do FRED](https://fred.stlouisfed.org/series/PFOODINDEXM).
- Datas dos eventos de 2026 (guerra do Irã, Estreito de Ormuz, reajuste do diesel):
  [Banco Mundial](https://blogs.worldbank.org/en/opendata/strait-of-hormuz-disruption-sends-oil-prices-surging),
  [CNBC](https://www.cnbc.com/2026/04/21/oil-price-iran-war-middle-east.html) e
  [Diário de Pernambuco](https://www.diariodepernambuco.com.br/economia/2026/03/11709805-petrobras-reajusta-preco-do-diesel-nas-refinarias-em-116-para-rs-365-por-litro.html).

As séries de referência somam cerca de 350 KB e ficam versionadas em `referencias/`, para o pipeline
rodar igual na máquina local, no CI e no Databricks. Cinco vêm como saíram das fontes; a da ANP é a
planilha convertida em CSV por `scripts/baixar_referencias.py`, porque o Spark não lê `.xlsx`. O
script confere o conteúdo com as regras do pipeline antes de gravar qualquer arquivo.
