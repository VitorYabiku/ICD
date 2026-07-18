import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

EP3_DIRECTORY_PATH: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIRECTORY_PATH: Final[Path] = EP3_DIRECTORY_PATH.parent
DATASET_DIRECTORY_PATH: Final[Path] = PROJECT_DIRECTORY_PATH / "dataset"
DATASET_PATH: Final[Path] = DATASET_DIRECTORY_PATH / "housing_stratified.csv"
OUTPUT_DIRECTORY_PATH: Final[Path] = EP3_DIRECTORY_PATH / "output"
OUTPUT_PATH: Final[Path] = OUTPUT_DIRECTORY_PATH / "resultados.json"

CROSS_VALIDATION_FOLD_COUNT: Final[int] = 10
RANDOM_STATE: Final[int] = 42
CROSS_VALIDATION_SCORING: Final[str] = "r2"
GRID_SEARCH_N_JOBS: Final[int] = -1
RANDOM_FOREST_N_JOBS: Final[int] = 1
DECISION_TREE_MAX_DEPTH_VALUES: Final = (None, 3, 5, 10, 20)
DECISION_TREE_MIN_SAMPLES_SPLIT_VALUES: Final = (2, 5, 10)
DECISION_TREE_MIN_SAMPLES_LEAF_VALUES: Final = (1, 2, 4)
RANDOM_FOREST_N_ESTIMATORS_VALUES: Final = (50, 100, 200, 400)
RANDOM_FOREST_MAX_DEPTH_VALUES: Final = (None, 5, 10, 20)
RANDOM_FOREST_MAX_FEATURES_VALUES: Final = (1.0, "sqrt", "log2")
OUTLIER_INTERQUARTILE_RANGE_FACTOR: Final[float] = 1.5
DECISION_TREE_PARAM_GRID: Final = {
    "max_depth": DECISION_TREE_MAX_DEPTH_VALUES,
    "min_samples_split": DECISION_TREE_MIN_SAMPLES_SPLIT_VALUES,
    "min_samples_leaf": DECISION_TREE_MIN_SAMPLES_LEAF_VALUES,
}
RANDOM_FOREST_PARAM_GRID: Final = {
    "n_estimators": RANDOM_FOREST_N_ESTIMATORS_VALUES,
    "max_depth": RANDOM_FOREST_MAX_DEPTH_VALUES,
    "max_features": RANDOM_FOREST_MAX_FEATURES_VALUES,
}
type Hyperparameters = dict[str, object]
type EvaluationScores = dict[str, list[float]]
type HyperparameterSelection = dict[str, object]

TARGET_VARIABLE_COLUMN_NAME: Final = "median_income"
CLUSTER_COLUMN_NAME: Final = "cluster"
TARGET_VARIABLE_MEASUREMENT_UNIT: Final = (
    "dezenas de milhares de dólares americanos; 1 unidade = US$ 10.000"
)
NUMERIC_FEATURE_COLUMN_NAMES: Final = (
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
)
OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING: Final = (
    "ISLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "<1H OCEAN",
    "INLAND",
)
LINEAR_REGRESSION_NUMERIC_FEATURE_INDICES: Final = tuple(
    range(len(NUMERIC_FEATURE_COLUMN_NAMES))
)
INNER_CROSS_VALIDATION: Final = StratifiedKFold(
    n_splits=CROSS_VALIDATION_FOLD_COUNT,
    shuffle=True,
    random_state=RANDOM_STATE,
)
OUTER_CROSS_VALIDATION: Final = StratifiedKFold(
    n_splits=CROSS_VALIDATION_FOLD_COUNT,
    shuffle=True,
    random_state=RANDOM_STATE,
)

ROW_WITH_NULL_OR_NAN_EXPR: Final[pl.Expr] = pl.any_horizontal(
    pl.all().is_null()
) | pl.any_horizontal(cs.float().is_nan())


def outlier_expr(column_name: str) -> pl.Expr:
    first_quartile = pl.col(column_name).quantile(0.25)
    third_quartile = pl.col(column_name).quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    return (
        pl.col(column_name)
        < first_quartile - OUTLIER_INTERQUARTILE_RANGE_FACTOR * interquartile_range
    ) | (
        pl.col(column_name)
        > third_quartile + OUTLIER_INTERQUARTILE_RANGE_FACTOR * interquartile_range
    )


ROW_WITH_OUTLIER_EXPR: Final[pl.Expr] = pl.any_horizontal(
    *[
        outlier_expr(column_name)
        for column_name in (TARGET_VARIABLE_COLUMN_NAME, *NUMERIC_FEATURE_COLUMN_NAMES)
    ]
)


def features_target_and_clusters_get(
    dataset: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = dataset.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    clusters = dataset.get_column(CLUSTER_COLUMN_NAME).to_numpy()
    features = dataset.drop(TARGET_VARIABLE_COLUMN_NAME, CLUSTER_COLUMN_NAME).to_numpy()
    return features, target, clusters


def error_score_format(score: float, target_mean: float) -> str:
    return f"{score:.4f} ({score / target_mean:.2%} da média da coluna alvo)"


def evaluation_result_get(
    scores: EvaluationScores, target_mean: float
) -> dict[str, object]:
    folds = [
        {
            "fold": fold,
            "r2": r2,
            "rmse": {"valor": rmse, "proporcao_media_alvo": rmse / target_mean},
            "mae": {"valor": mae, "proporcao_media_alvo": mae / target_mean},
        }
        for fold, (r2, rmse, mae) in enumerate(
            zip(scores["r2"], scores["rmse"], scores["mae"], strict=True), start=1
        )
    ]
    mean_rmse = float(np.mean(scores["rmse"]))
    mean_mae = float(np.mean(scores["mae"]))
    return {
        "folds": folds,
        "medias": {
            "r2": float(np.mean(scores["r2"])),
            "rmse": {
                "valor": mean_rmse,
                "proporcao_media_alvo": mean_rmse / target_mean,
            },
            "mae": {
                "valor": mean_mae,
                "proporcao_media_alvo": mean_mae / target_mean,
            },
        },
    }


def model_evaluation_print(
    model_name: str, scores: EvaluationScores, target_mean: float
) -> None:
    print(f"\n--- Avaliação: {model_name} ---")
    print("\nAs métricas externas abaixo foram calculadas em cada fold externa.")
    for fold, (r2, rmse, mae) in enumerate(
        zip(scores["r2"], scores["rmse"], scores["mae"], strict=True), start=1
    ):
        print(
            f"Fold {fold}: R² = {r2:.4f}; "
            f"RMSE = {error_score_format(rmse, target_mean)}; "
            f"MAE = {error_score_format(mae, target_mean)}"
        )
    print(f"R² médio: {np.mean(scores['r2']):.4f}")
    for metric, label in (("rmse", "RMSE"), ("mae", "MAE")):
        mean_score = float(np.mean(scores[metric]))
        print(f"{label} médio: {error_score_format(mean_score, target_mean)}")


def linear_regression_train(
    dataset_lazy: pl.LazyFrame,
    target_mean: float,
) -> EvaluationScores:
    print("\n=== Regressão linear ===")
    print("--- Treinamento: Regressão linear ---")
    dataset = dataset_lazy.collect()
    features, target, clusters = features_target_and_clusters_get(dataset)
    model: Final = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            StandardScaler(),
                            LINEAR_REGRESSION_NUMERIC_FEATURE_INDICES,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    cross_validation_results = cross_validate(
        model,
        features,
        target,
        cv=OUTER_CROSS_VALIDATION.split(features, clusters),
        scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
    )
    scores: EvaluationScores = {
        "r2": [float(score) for score in cross_validation_results["test_r2"]],
        "rmse": [
            -float(score)
            for score in cross_validation_results["test_neg_root_mean_squared_error"]
        ],
        "mae": [
            -float(score)
            for score in cross_validation_results["test_neg_mean_absolute_error"]
        ],
    }

    model_evaluation_print("Regressão linear", scores, target_mean)

    return scores


def decision_tree_train(
    dataset_lazy: pl.LazyFrame,
    target_mean: float,
) -> tuple[EvaluationScores, Hyperparameters, HyperparameterSelection]:
    dataset = dataset_lazy.collect()
    features, target, clusters = features_target_and_clusters_get(dataset)
    print("\n=== Árvore de decisão ===")
    print("--- Treinamento e seleção de hiperparâmetros: Árvore de decisão ---")
    scores: EvaluationScores = {"r2": [], "rmse": [], "mae": []}
    candidate_mean_scores_by_fold: list[np.ndarray] = []
    best_candidate_indices: list[int] = []
    selection_by_fold: list[dict[str, object]] = []
    candidate_params: np.ndarray = np.array([], dtype=object)
    for fold, (train_indices, test_indices) in enumerate(
        OUTER_CROSS_VALIDATION.split(features, clusters), start=1
    ):
        print(f"Processando fold externa {fold}/{CROSS_VALIDATION_FOLD_COUNT}...")
        search = GridSearchCV(
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            DECISION_TREE_PARAM_GRID,
            cv=list(
                INNER_CROSS_VALIDATION.split(
                    features[train_indices], clusters[train_indices]
                )
            ),
            scoring=CROSS_VALIDATION_SCORING,
            n_jobs=GRID_SEARCH_N_JOBS,
        )
        search.fit(features[train_indices], target[train_indices])
        candidate_mean_scores_by_fold.append(search.cv_results_["mean_test_score"])
        best_candidate_indices.append(search.best_index_)
        candidate_params = np.asarray(search.cv_results_["params"], dtype=object)
        best_params = search.best_params_
        selection_by_fold.append(
            {
                "fold": fold,
                "melhores_hiperparametros": dict(best_params),
                "r2_medio_interno": search.best_score_,
            }
        )
        predictions = search.predict(features[test_indices])
        r2 = float(search.score(features[test_indices], target[test_indices]))
        rmse = root_mean_squared_error(target[test_indices], predictions)
        mae = mean_absolute_error(target[test_indices], predictions)
        scores["r2"].append(r2)
        scores["rmse"].append(rmse)
        scores["mae"].append(mae)
        print(
            f"Melhores hiperparâmetros da fold externa {fold}: "
            f"max_depth={best_params['max_depth']}, "
            f"min_samples_split={best_params['min_samples_split']}, "
            f"min_samples_leaf={best_params['min_samples_leaf']}; "
            f"R² médio interno={search.best_score_:.4f}; "
            f"R² externo={r2:.4f}; "
            f"RMSE externo={error_score_format(rmse, target_mean)}; "
            f"MAE externo={error_score_format(mae, target_mean)}"
        )

    overall_candidate_mean_scores = np.mean(candidate_mean_scores_by_fold, axis=0)
    overall_best_candidate_index = int(np.argmax(overall_candidate_mean_scores))
    overall_best_params = candidate_params[overall_best_candidate_index]
    overall_best_folds = ", ".join(
        str(fold)
        for fold, best_candidate_index in enumerate(best_candidate_indices, start=1)
        if best_candidate_index == overall_best_candidate_index
    )
    print(
        "\nOs R² externos abaixo pertencem à melhor combinação de cada fold "
        "externa, não a uma única combinação geral."
    )
    model_evaluation_print("Árvore de decisão", scores, target_mean)
    print(
        "Melhor combinação geral por R² médio interno: "
        f"max_depth={overall_best_params['max_depth']}, "
        f"min_samples_split={overall_best_params['min_samples_split']}, "
        f"min_samples_leaf={overall_best_params['min_samples_leaf']}; "
        "R² médio interno entre folds externas="
        f"{overall_candidate_mean_scores[overall_best_candidate_index]:.4f}; "
        f"melhor nas folds externas: {overall_best_folds or 'nenhuma'}"
    )

    return (
        scores,
        dict(overall_best_params),
        {
            "folds": selection_by_fold,
            "melhores_hiperparametros_gerais": dict(overall_best_params),
            "r2_medio_interno_geral": float(
                overall_candidate_mean_scores[overall_best_candidate_index]
            ),
            "folds_com_melhor_combinacao_geral": [
                fold
                for fold, best_candidate_index in enumerate(
                    best_candidate_indices, start=1
                )
                if best_candidate_index == overall_best_candidate_index
            ],
        },
    )


def random_forest_train(
    dataset_lazy: pl.LazyFrame,
    target_mean: float,
) -> tuple[EvaluationScores, Hyperparameters, HyperparameterSelection]:
    dataset = dataset_lazy.collect()
    features, target, clusters = features_target_and_clusters_get(dataset)
    print("\n=== Floresta aleatória ===")
    print("--- Treinamento e seleção de hiperparâmetros: Floresta aleatória ---")
    scores: EvaluationScores = {"r2": [], "rmse": [], "mae": []}
    candidate_mean_scores_by_fold: list[np.ndarray] = []
    best_candidate_indices: list[int] = []
    selection_by_fold: list[dict[str, object]] = []
    candidate_params: np.ndarray = np.array([], dtype=object)
    for fold, (train_indices, test_indices) in enumerate(
        OUTER_CROSS_VALIDATION.split(features, clusters), start=1
    ):
        print(f"Processando fold externa {fold}/{CROSS_VALIDATION_FOLD_COUNT}...")
        search = GridSearchCV(
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=RANDOM_FOREST_N_JOBS,
            ),
            RANDOM_FOREST_PARAM_GRID,
            cv=list(
                INNER_CROSS_VALIDATION.split(
                    features[train_indices], clusters[train_indices]
                )
            ),
            scoring=CROSS_VALIDATION_SCORING,
            n_jobs=GRID_SEARCH_N_JOBS,
        )
        search.fit(features[train_indices], target[train_indices])
        candidate_mean_scores_by_fold.append(search.cv_results_["mean_test_score"])
        best_candidate_indices.append(search.best_index_)
        candidate_params = np.asarray(search.cv_results_["params"], dtype=object)
        best_params = search.best_params_
        selection_by_fold.append(
            {
                "fold": fold,
                "melhores_hiperparametros": dict(best_params),
                "r2_medio_interno": search.best_score_,
            }
        )
        predictions = search.predict(features[test_indices])
        r2 = float(search.score(features[test_indices], target[test_indices]))
        rmse = root_mean_squared_error(target[test_indices], predictions)
        mae = mean_absolute_error(target[test_indices], predictions)
        scores["r2"].append(r2)
        scores["rmse"].append(rmse)
        scores["mae"].append(mae)
        print(
            f"Melhores hiperparâmetros da fold externa {fold}: "
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, "
            f"max_features={best_params['max_features']}; "
            f"R² médio interno={search.best_score_:.4f}; "
            f"R² externo={r2:.4f}; "
            f"RMSE externo={error_score_format(rmse, target_mean)}; "
            f"MAE externo={error_score_format(mae, target_mean)}"
        )

    overall_candidate_mean_scores = np.mean(candidate_mean_scores_by_fold, axis=0)
    overall_best_candidate_index = int(np.argmax(overall_candidate_mean_scores))
    overall_best_params = candidate_params[overall_best_candidate_index]
    overall_best_folds = ", ".join(
        str(fold)
        for fold, best_candidate_index in enumerate(best_candidate_indices, start=1)
        if best_candidate_index == overall_best_candidate_index
    )
    model_evaluation_print("Floresta aleatória", scores, target_mean)
    print(
        "Melhor combinação geral por R² médio interno: "
        f"n_estimators={overall_best_params['n_estimators']}, "
        f"max_depth={overall_best_params['max_depth']}, "
        f"max_features={overall_best_params['max_features']}; "
        "R² médio interno entre folds externas="
        f"{overall_candidate_mean_scores[overall_best_candidate_index]:.4f}; "
        f"melhor nas folds externas: {overall_best_folds or 'nenhuma'}"
    )

    return (
        scores,
        dict(overall_best_params),
        {
            "folds": selection_by_fold,
            "melhores_hiperparametros_gerais": dict(overall_best_params),
            "r2_medio_interno_geral": float(
                overall_candidate_mean_scores[overall_best_candidate_index]
            ),
            "folds_com_melhor_combinacao_geral": [
                fold
                for fold, best_candidate_index in enumerate(
                    best_candidate_indices, start=1
                )
                if best_candidate_index == overall_best_candidate_index
            ],
        },
    )


def models_comparison_print(
    model_scores: list[tuple[str, EvaluationScores]],
    target_mean: float,
) -> None:
    print("\n=== Comparação final ===")
    for model_name, scores in sorted(
        model_scores, key=lambda item: np.mean(item[1]["r2"]), reverse=True
    ):
        mean_rmse = float(np.mean(scores["rmse"]))
        mean_mae = float(np.mean(scores["mae"]))
        print(
            f"{model_name}: "
            f"R² médio={np.mean(scores['r2']):.4f}; "
            f"RMSE médio={error_score_format(mean_rmse, target_mean)}; "
            f"MAE médio={error_score_format(mean_mae, target_mean)}"
        )


def hyperparameter_comparison_print(
    best_hyperparameters_by_dataset: dict[str, dict[str, Hyperparameters]],
) -> None:
    print("\n=== Sensibilidade dos hiperparâmetros a outliers ===")
    print("Legenda: com outliers → sem outliers; = igual; ≠ diferente")
    with_outliers = best_hyperparameters_by_dataset["Dados com outliers"]
    without_outliers = best_hyperparameters_by_dataset["Dados sem outliers"]
    for model_name, model_hyperparameters_with_outliers in with_outliers.items():
        print(f"{model_name}:")
        model_hyperparameters_without_outliers = without_outliers[model_name]
        for (
            hyperparameter_name,
            value_with_outliers,
        ) in model_hyperparameters_with_outliers.items():
            value_without_outliers = model_hyperparameters_without_outliers[
                hyperparameter_name
            ]
            symbol = "=" if value_with_outliers == value_without_outliers else "≠"
            print(
                f"  {hyperparameter_name}: {value_with_outliers} → "
                f"{value_without_outliers} {symbol}"
            )


def main() -> None:
    dataset_lazy: pl.LazyFrame = pl.scan_csv(
        DATASET_PATH,
        schema_overrides={
            "ocean_proximity": pl.Enum(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING)
        },
    )

    total_row_count = dataset_lazy.select(pl.len()).collect().item()
    invalid_row_count = (
        dataset_lazy.filter(ROW_WITH_NULL_OR_NAN_EXPR).select(pl.len()).collect().item()
    )
    print("=== Preparação dos dados ===")
    print(f"Linhas totais: {total_row_count}")
    print(f"Linhas removidas por null ou NaN: {invalid_row_count}")
    print(f"Linhas utilizadas: {total_row_count - invalid_row_count}")

    dataset = (
        dataset_lazy.filter(~ROW_WITH_NULL_OR_NAN_EXPR)
        .select(
            TARGET_VARIABLE_COLUMN_NAME,
            (pl.col("total_rooms") / pl.col("households")).alias("rooms_per_household"),
            (pl.col("total_bedrooms") / pl.col("total_rooms")).alias(
                "bedrooms_per_room"
            ),
            (pl.col("population") / pl.col("households")).alias(
                "population_per_household"
            ),
            "ocean_proximity",
            CLUSTER_COLUMN_NAME,
        )
        .collect()
    )
    dataset_without_outliers = dataset.filter(~ROW_WITH_OUTLIER_EXPR)
    outlier_row_count = dataset.height - dataset_without_outliers.height
    print(
        "Linhas removidas por um ou mais outliers: "
        f"{outlier_row_count} ({outlier_row_count / dataset.height:.2%})"
    )
    print(f"Linhas utilizadas sem outliers: {dataset_without_outliers.height}")

    results: dict[str, object] = {
        "preparacao_dos_dados": {
            "linhas_totais": total_row_count,
            "linhas_removidas_por_nulo_ou_nan": invalid_row_count,
            "linhas_utilizadas": total_row_count - invalid_row_count,
            "linhas_removidas_por_outliers": {
                "quantidade": outlier_row_count,
                "proporcao": outlier_row_count / dataset.height,
            },
            "linhas_utilizadas_sem_outliers": dataset_without_outliers.height,
        }
    }
    datasets_results: dict[str, object] = {}
    mean_scores_by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    target_means_by_dataset: dict[str, float] = {}
    best_hyperparameters_by_dataset: dict[str, dict[str, Hyperparameters]] = {}
    for dataset_key, dataset_name, evaluation_dataset in (
        ("dados_com_outliers", "Dados com outliers", dataset),
        ("dados_sem_outliers", "Dados sem outliers", dataset_without_outliers),
    ):
        print(f"\n\n######## {dataset_name} ########")
        target = evaluation_dataset.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        target_mean = float(np.mean(target))
        target_means_by_dataset[dataset_name] = target_mean
        print(
            f"Escala de {TARGET_VARIABLE_COLUMN_NAME} "
            f"({TARGET_VARIABLE_MEASUREMENT_UNIT}): "
            f"mínimo={np.min(target):.4f}; média={target_mean:.4f}; "
            f"máximo={np.max(target):.4f}"
        )
        print("RMSE e MAE estão na mesma unidade da coluna alvo.")
        linear_regression_dataset_lazy = evaluation_dataset.to_dummies(
            "ocean_proximity",
            drop_first=True,  # Use dummy variables for linear regression
        ).lazy()
        linear_regression_scores = linear_regression_train(
            linear_regression_dataset_lazy, target_mean
        )

        tree_based_models_dataset_lazy = evaluation_dataset.to_dummies(
            "ocean_proximity",
            drop_first=False,  # Use one-hot encoding for tree-based models
        ).lazy()
        (
            decision_tree_scores,
            decision_tree_best_hyperparameters,
            decision_tree_hyperparameter_selection,
        ) = decision_tree_train(tree_based_models_dataset_lazy, target_mean)
        (
            random_forest_scores,
            random_forest_best_hyperparameters,
            random_forest_hyperparameter_selection,
        ) = random_forest_train(tree_based_models_dataset_lazy, target_mean)
        model_scores = [
            ("Regressão linear", linear_regression_scores),
            ("Árvore de decisão", decision_tree_scores),
            ("Floresta aleatória", random_forest_scores),
        ]
        models_comparison_print(model_scores, target_mean)
        mean_scores_by_dataset[dataset_name] = {
            model_name: {
                metric: float(np.mean(metric_scores))
                for metric, metric_scores in scores.items()
            }
            for model_name, scores in model_scores
        }
        best_hyperparameters_by_dataset[dataset_name] = {
            "Árvore de decisão": decision_tree_best_hyperparameters,
            "Floresta aleatória": random_forest_best_hyperparameters,
        }
        models_results = {
            "regressao_linear": {
                "nome": "Regressão linear",
                "avaliacao": evaluation_result_get(
                    linear_regression_scores, target_mean
                ),
            },
            "arvore_de_decisao": {
                "nome": "Árvore de decisão",
                "avaliacao": evaluation_result_get(decision_tree_scores, target_mean),
                "selecao_de_hiperparametros": decision_tree_hyperparameter_selection,
            },
            "floresta_aleatoria": {
                "nome": "Floresta aleatória",
                "avaliacao": evaluation_result_get(random_forest_scores, target_mean),
                "selecao_de_hiperparametros": random_forest_hyperparameter_selection,
            },
        }
        datasets_results[dataset_key] = {
            "nome": dataset_name,
            "quantidade_de_linhas": evaluation_dataset.height,
            "variavel_alvo": {
                "nome": TARGET_VARIABLE_COLUMN_NAME,
                "unidade_de_medida": TARGET_VARIABLE_MEASUREMENT_UNIT,
                "minimo": float(np.min(target)),
                "media": target_mean,
                "maximo": float(np.max(target)),
            },
            "modelos": models_results,
            "comparacao_dos_modelos": [
                {
                    "posicao": position,
                    "modelo": model_name,
                    "metricas_medias": dict(
                        mean_scores_by_dataset[dataset_name][model_name]
                    ),
                }
                for position, (model_name, _) in enumerate(
                    sorted(
                        model_scores,
                        key=lambda item: np.mean(item[1]["r2"]),
                        reverse=True,
                    ),
                    start=1,
                )
            ],
        }

    results["conjuntos_de_dados"] = datasets_results

    print("\n=== Sensibilidade a outliers ===")
    for model_name, scores_with_outliers in mean_scores_by_dataset[
        "Dados com outliers"
    ].items():
        scores_without_outliers = mean_scores_by_dataset["Dados sem outliers"][
            model_name
        ]
        print(f"{model_name}:")
        for metric, label in (("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")):
            score_with_outliers = scores_with_outliers[metric]
            score_without_outliers = scores_without_outliers[metric]
            normalized_difference_display = ""
            if metric == "r2":
                score_with_outliers_display = f"{score_with_outliers:.4f}"
                score_without_outliers_display = f"{score_without_outliers:.4f}"
            else:
                target_mean_with_outliers = target_means_by_dataset[
                    "Dados com outliers"
                ]
                target_mean_without_outliers = target_means_by_dataset[
                    "Dados sem outliers"
                ]
                score_with_outliers_display = error_score_format(
                    score_with_outliers,
                    target_mean_with_outliers,
                )
                score_without_outliers_display = error_score_format(
                    score_without_outliers,
                    target_mean_without_outliers,
                )
                normalized_difference = (
                    score_without_outliers / target_mean_without_outliers
                    - score_with_outliers / target_mean_with_outliers
                )
                normalized_difference_display = f""" ({normalized_difference * 100:+.2f}
                    p.p. da média da coluna alvo)"""
            print(
                f"  {label} médio com outliers={score_with_outliers_display}; "
                f"sem outliers={score_without_outliers_display}; "
                f"diferença={score_without_outliers - score_with_outliers:+.4f}"
                f"{normalized_difference_display}"
            )

    hyperparameter_comparison_print(best_hyperparameters_by_dataset)

    metric_sensitivity: dict[str, object] = {}
    for model_name, scores_with_outliers in mean_scores_by_dataset[
        "Dados com outliers"
    ].items():
        scores_without_outliers = mean_scores_by_dataset["Dados sem outliers"][
            model_name
        ]
        model_metric_sensitivity: dict[str, object] = {}
        for metric in ("r2", "rmse", "mae"):
            score_with_outliers = scores_with_outliers[metric]
            score_without_outliers = scores_without_outliers[metric]
            comparison: dict[str, object] = {
                "com_outliers": score_with_outliers,
                "sem_outliers": score_without_outliers,
                "diferenca": score_without_outliers - score_with_outliers,
            }
            if metric != "r2":
                comparison["diferenca_normalizada_pela_media_da_variavel_alvo"] = (
                    score_without_outliers
                    / target_means_by_dataset["Dados sem outliers"]
                    - score_with_outliers
                    / target_means_by_dataset["Dados com outliers"]
                )
            model_metric_sensitivity[metric] = comparison
        metric_sensitivity[model_name] = model_metric_sensitivity

    hyperparameter_sensitivity: dict[str, object] = {}
    for model_name, hyperparameters_with_outliers in best_hyperparameters_by_dataset[
        "Dados com outliers"
    ].items():
        hyperparameters_without_outliers = best_hyperparameters_by_dataset[
            "Dados sem outliers"
        ][model_name]
        hyperparameter_sensitivity[model_name] = {
            name: {
                "com_outliers": value_with_outliers,
                "sem_outliers": hyperparameters_without_outliers[name],
                "igual": value_with_outliers == hyperparameters_without_outliers[name],
            }
            for name, value_with_outliers in hyperparameters_with_outliers.items()
        }

    results["sensibilidade_a_outliers"] = {
        "metricas": metric_sensitivity,
        "hiperparametros": hyperparameter_sensitivity,
    }
    OUTPUT_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)
    temporary_output_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_output_path.replace(OUTPUT_PATH)
    print(f"\nResultados salvos em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
