from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

EP3_DIRECTORY_PATH: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIRECTORY_PATH: Final[Path] = EP3_DIRECTORY_PATH.parent
DATASET_DIRECTORY_PATH: Final[Path] = PROJECT_DIRECTORY_PATH / "dataset"
DATASET_PATH: Final[Path] = DATASET_DIRECTORY_PATH / "housing.csv"

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

TARGET_VARIABLE_COLUMN_NAME: Final = "median_income"
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
INNER_CROSS_VALIDATION: Final = KFold(
    n_splits=CROSS_VALIDATION_FOLD_COUNT,
    shuffle=True,
    random_state=RANDOM_STATE,
)
OUTER_CROSS_VALIDATION: Final = KFold(
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


def features_and_target_get(dataset: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target = dataset.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    features = dataset.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    return features, target


def model_evaluation_print(model_name: str, scores: EvaluationScores) -> None:
    print(f"\n--- Avaliação: {model_name} ---")
    print("\nAs métricas externas abaixo foram calculadas em cada fold externa.")
    for fold, (r2, rmse, mae) in enumerate(
        zip(scores["r2"], scores["rmse"], scores["mae"], strict=True), start=1
    ):
        print(f"Fold {fold}: R² = {r2:.4f}; RMSE = {rmse:.4f}; MAE = {mae:.4f}")
    for metric, label in (("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")):
        print(f"{label} médio: {np.mean(scores[metric]):.4f}")


def linear_regression_train(
    dataset_lazy: pl.LazyFrame,
) -> EvaluationScores:
    print("\n=== Regressão linear ===")
    print("--- Treinamento: Regressão linear ---")
    dataset = dataset_lazy.collect()
    features, target = features_and_target_get(dataset)
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
        cv=OUTER_CROSS_VALIDATION,
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

    model_evaluation_print("Regressão linear", scores)

    return scores


def decision_tree_train(
    dataset_lazy: pl.LazyFrame,
) -> tuple[EvaluationScores, Hyperparameters]:
    dataset = dataset_lazy.collect()
    features, target = features_and_target_get(dataset)
    print("\n=== Árvore de decisão ===")
    print("--- Treinamento e seleção de hiperparâmetros: Árvore de decisão ---")
    scores: EvaluationScores = {"r2": [], "rmse": [], "mae": []}
    candidate_mean_scores_by_fold: list[np.ndarray] = []
    best_candidate_indices: list[int] = []
    candidate_params: np.ndarray = np.array([], dtype=object)
    for fold, (train_indices, test_indices) in enumerate(
        OUTER_CROSS_VALIDATION.split(features), start=1
    ):
        print(f"Processando fold externa {fold}/{CROSS_VALIDATION_FOLD_COUNT}...")
        search = GridSearchCV(
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            DECISION_TREE_PARAM_GRID,
            cv=INNER_CROSS_VALIDATION,
            scoring=CROSS_VALIDATION_SCORING,
            n_jobs=GRID_SEARCH_N_JOBS,
        )
        search.fit(features[train_indices], target[train_indices])
        candidate_mean_scores_by_fold.append(search.cv_results_["mean_test_score"])
        best_candidate_indices.append(search.best_index_)
        candidate_params = np.asarray(search.cv_results_["params"], dtype=object)
        best_params = search.best_params_
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
            f"RMSE externo={rmse:.4f}; "
            f"MAE externo={mae:.4f}"
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
    model_evaluation_print("Árvore de decisão", scores)
    print(
        "Melhor combinação geral por R² médio interno: "
        f"max_depth={overall_best_params['max_depth']}, "
        f"min_samples_split={overall_best_params['min_samples_split']}, "
        f"min_samples_leaf={overall_best_params['min_samples_leaf']}; "
        "R² médio interno entre folds externas="
        f"{overall_candidate_mean_scores[overall_best_candidate_index]:.4f}; "
        f"melhor nas folds externas: {overall_best_folds or 'nenhuma'}"
    )

    return scores, dict(overall_best_params)


def random_forest_train(
    dataset_lazy: pl.LazyFrame,
) -> tuple[EvaluationScores, Hyperparameters]:
    dataset = dataset_lazy.collect()
    features, target = features_and_target_get(dataset)
    print("\n=== Floresta aleatória ===")
    print("--- Treinamento e seleção de hiperparâmetros: Floresta aleatória ---")
    scores: EvaluationScores = {"r2": [], "rmse": [], "mae": []}
    candidate_mean_scores_by_fold: list[np.ndarray] = []
    best_candidate_indices: list[int] = []
    candidate_params: np.ndarray = np.array([], dtype=object)
    for fold, (train_indices, test_indices) in enumerate(
        OUTER_CROSS_VALIDATION.split(features), start=1
    ):
        print(f"Processando fold externa {fold}/{CROSS_VALIDATION_FOLD_COUNT}...")
        search = GridSearchCV(
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=RANDOM_FOREST_N_JOBS,
            ),
            RANDOM_FOREST_PARAM_GRID,
            cv=INNER_CROSS_VALIDATION,
            scoring=CROSS_VALIDATION_SCORING,
            n_jobs=GRID_SEARCH_N_JOBS,
        )
        search.fit(features[train_indices], target[train_indices])
        candidate_mean_scores_by_fold.append(search.cv_results_["mean_test_score"])
        best_candidate_indices.append(search.best_index_)
        candidate_params = np.asarray(search.cv_results_["params"], dtype=object)
        best_params = search.best_params_
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
            f"RMSE externo={rmse:.4f}; "
            f"MAE externo={mae:.4f}"
        )

    overall_candidate_mean_scores = np.mean(candidate_mean_scores_by_fold, axis=0)
    overall_best_candidate_index = int(np.argmax(overall_candidate_mean_scores))
    overall_best_params = candidate_params[overall_best_candidate_index]
    overall_best_folds = ", ".join(
        str(fold)
        for fold, best_candidate_index in enumerate(best_candidate_indices, start=1)
        if best_candidate_index == overall_best_candidate_index
    )
    model_evaluation_print("Floresta aleatória", scores)
    print(
        "Melhor combinação geral por R² médio interno: "
        f"n_estimators={overall_best_params['n_estimators']}, "
        f"max_depth={overall_best_params['max_depth']}, "
        f"max_features={overall_best_params['max_features']}; "
        "R² médio interno entre folds externas="
        f"{overall_candidate_mean_scores[overall_best_candidate_index]:.4f}; "
        f"melhor nas folds externas: {overall_best_folds or 'nenhuma'}"
    )

    return scores, dict(overall_best_params)


def models_comparison_print(
    model_scores: list[tuple[str, EvaluationScores]],
) -> None:
    print("\n=== Comparação final ===")
    for model_name, scores in sorted(
        model_scores, key=lambda item: np.mean(item[1]["r2"]), reverse=True
    ):
        print(
            f"{model_name}: "
            f"R² médio={np.mean(scores['r2']):.4f}; "
            f"RMSE médio={np.mean(scores['rmse']):.4f}; "
            f"MAE médio={np.mean(scores['mae']):.4f}"
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

    mean_scores_by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    best_hyperparameters_by_dataset: dict[str, dict[str, Hyperparameters]] = {}
    for dataset_name, evaluation_dataset in (
        ("Dados com outliers", dataset),
        ("Dados sem outliers", dataset_without_outliers),
    ):
        print(f"\n\n######## {dataset_name} ########")
        linear_regression_dataset_lazy = evaluation_dataset.to_dummies(
            "ocean_proximity",
            drop_first=True,  # Use dummy variables for linear regression
        ).lazy()
        linear_regression_scores = linear_regression_train(
            linear_regression_dataset_lazy
        )

        tree_based_models_dataset_lazy = evaluation_dataset.to_dummies(
            "ocean_proximity",
            drop_first=False,  # Use one-hot encoding for tree-based models
        ).lazy()
        decision_tree_scores, decision_tree_best_hyperparameters = decision_tree_train(
            tree_based_models_dataset_lazy
        )
        random_forest_scores, random_forest_best_hyperparameters = random_forest_train(
            tree_based_models_dataset_lazy
        )
        model_scores = [
            ("Regressão linear", linear_regression_scores),
            ("Árvore de decisão", decision_tree_scores),
            ("Floresta aleatória", random_forest_scores),
        ]
        models_comparison_print(model_scores)
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
            print(
                f"  {label} médio com outliers={score_with_outliers:.4f}; "
                f"sem outliers={score_without_outliers:.4f}; "
                f"diferença={score_without_outliers - score_with_outliers:+.4f}"
            )

    hyperparameter_comparison_print(best_hyperparameters_by_dataset)


if __name__ == "__main__":
    main()
