import json
from pathlib import Path
from typing import Final

import seaborn as sns

from utils import subplots


def main() -> None:
    EP3_DIRECTORY_PATH: Final = Path(__file__).resolve().parent
    RESULTS_PATH: Final = EP3_DIRECTORY_PATH / "output" / "resultados.json"
    PLOT_PATH: Final = EP3_DIRECTORY_PATH / "output" / "r2_modelos.eps"

    MODEL_NAMES: Final = {
        "regressao_linear": "Regressão linear múltipla",
        "arvore_de_decisao": "Árvore de decisão",
        "floresta_aleatoria": "Floresta aleatória",
    }
    DATASET_NAMES: Final = {
        "dados_com_outliers": "Com outliers",
        "dados_sem_outliers": "Sem outliers",
    }

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    model_names: list[str] = []
    dataset_names: list[str] = []
    r2_values: list[float] = []

    for model_key, model_name in MODEL_NAMES.items():
        for dataset_key, dataset_name in DATASET_NAMES.items():
            model_names.append(model_name)
            dataset_names.append(dataset_name)
            r2_values.append(
                results["conjuntos_de_dados"][dataset_key]["modelos"][model_key][
                    "avaliacao"
                ]["medias"]["r2"]
            )

    plot_data = {
        "Modelo": model_names,
        "Conjunto de dados": dataset_names,
        "R²": r2_values,
    }
    sns.set_theme()
    with subplots(savefig_path=PLOT_PATH) as axis:
        sns.barplot(
            data=plot_data,
            x="Modelo",
            y="R²",
            hue="Conjunto de dados",
            # Values are already fold averages, so no confidence interval is needed.
            errorbar=None,
            ax=axis,
        )
        # EPS does not support the default translucent legend background.
        axis.legend(framealpha=1)


if __name__ == "__main__":
    main()
