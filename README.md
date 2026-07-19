Precisa instalar o [uv](https://docs.astral.sh/uv/) antes de rodar os scripts.

Rodar scripts a partir da raiz do repo:

```sh
uv run -m ep2.clustering_for_sampling
uv run -m ep2.main
uv run -m ep2.scripts.json_tables_to_latex

uv run -m ep3.main
uv run -m ep3.analysis
```

saídas:

- ep2/clustering_for_sampling.py
  - ep2/clustering_outputs/median_income_{single,complete,average,ward}_clusterizacao.png
  - ep2/clustering_outputs/median_income_best_clusterizacao_resumo.png
  - dataset/housing_stratified.csv

- ep2/main.py
  - ep2/plots/*.png
  - ep2/tables/null_or_nan_estatisticas_descritivas_*.json
  - ep2/tables/estatisticas_descritivas_*.json
  - ep2/tables/dados_transformados_estatisticas_descritivas_*.json

- ep2/scripts/json_tables_to_latex.py
  - ep2/tables_latex/null_or_nan_estatisticas_descritivas.tex
  - ep2/tables_latex/estatisticas_descritivas.tex
  - ep2/tables_latex/dados_transformados_estatisticas_descritivas.tex

- ep3/main.py
  - ep3/output/resultados.json
  - ep3/output/modelos/{dados_com_outliers,dados_sem_outliers}_{regressao_linear,arvore_de_decisao,floresta_aleatoria}.joblib

- ep3/analysis.py
  - ep3/output/r2_modelos.eps
  - ep3/output/importancia_features_{regressao_linear,arvore_de_decisao,floresta_aleatoria}.eps
