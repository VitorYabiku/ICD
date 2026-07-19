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
        "regressao_linear": "Regressão linear",
        "arvore_de_decisao": "Árvore de decisão",
        "floresta_aleatoria": "Floresta aleatória",
    }
    DATASET_NAMES: Final = {
        "dados_com_outliers": "Com outliers",
        "dados_sem_outliers": "Sem outliers",
    }
    FEATURE_IMPORTANCE_METHOD_DESCRIPTIONS: Final = {
        "regressao_linear": (
            "Importância = média do valor absoluto dos coeficientes nos folds"
        ),
        "arvore_de_decisao": (
            "Importância = média da redução normalizada da impureza nos folds"
        ),
        "floresta_aleatoria": (
            "Importância = média da redução normalizada da impureza nos folds"
        ),
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
        for container in axis.containers:
            axis.bar_label(container, fmt="%.4f", padding=3)
        # EPS does not support the default translucent legend background.
        axis.set_ylim(0, 1)
        axis.set_title("Comparação do R² dos modelos por conjunto de dados")
        axis.legend(framealpha=1)

    for model_key, model_name in MODEL_NAMES.items():
        feature_names: list[str] = []
        feature_dataset_names: list[str] = []
        importance_values: list[float] = []
        importance_values_by_feature: dict[str, list[float]] = {}
        for dataset_key, dataset_name in DATASET_NAMES.items():
            features = results["conjuntos_de_dados"][dataset_key]["modelos"][
                model_key
            ]["importancia_das_features"]["media_geral"]
            for feature in features:
                feature_name = feature["feature"]
                importance = feature["importancia"]
                feature_names.append(feature_name)
                feature_dataset_names.append(dataset_name)
                importance_values.append(importance)
                importance_values_by_feature.setdefault(feature_name, []).append(
                    importance
                )

        feature_order = sorted(
            importance_values_by_feature,
            key=lambda feature_name: sum(
                importance_values_by_feature[feature_name]
            )
            / len(importance_values_by_feature[feature_name]),
            reverse=True,
        )
        plot_data = {
            "Feature": feature_names,
            "Conjunto de dados": feature_dataset_names,
            "Importância": importance_values,
        }
        plot_path = (
            EP3_DIRECTORY_PATH
            / "output"
            / f"importancia_features_{model_key}.eps"
        )
        with subplots(figsize=(10, 6), savefig_path=plot_path) as axis:
            sns.barplot(
                data=plot_data,
                x="Importância",
                y="Feature",
                hue="Conjunto de dados",
                order=feature_order,
                errorbar=None,
                ax=axis,
            )
            axis.set_title(
                f"Importância das features — {model_name}\n"
                f"{FEATURE_IMPORTANCE_METHOD_DESCRIPTIONS[model_key]}"
            )
            axis.legend(framealpha=1)


if __name__ == "__main__":
    main()
