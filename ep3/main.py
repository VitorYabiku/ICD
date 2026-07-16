from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

EP3_DIRECTORY_PATH: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIRECTORY_PATH: Final[Path] = EP3_DIRECTORY_PATH.parent
DATASET_DIRECTORY_PATH: Final[Path] = PROJECT_DIRECTORY_PATH / "dataset"
DATASET_PATH: Final[Path] = DATASET_DIRECTORY_PATH / "housing.csv"

CROSS_VALIDATION_FOLD_COUNT: Final[int] = 5
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


def features_and_target_get(dataset: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target = dataset.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    features = dataset.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    return features, target


def model_evaluation_print(model_name: str, scores: list[float]) -> None:
    print(f"\n--- Avaliação: {model_name} ---")
    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: R² = {score:.4f}")
    print(f"R² médio: {np.mean(scores):.4f}")
    print(f"Desvio-padrão do R²: {np.std(scores):.4f}")


def linear_regression_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
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
                            MinMaxScaler(),
                            LINEAR_REGRESSION_NUMERIC_FEATURE_INDICES,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    scores = [
        float(score)
        for score in cross_val_score(
            model,
            features,
            target,
            cv=OUTER_CROSS_VALIDATION,
            scoring=CROSS_VALIDATION_SCORING,
        )
    ]

    model_evaluation_print("Regressão linear", scores)

    return scores


def decision_tree_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
    dataset = dataset_lazy.collect()
    features, target = features_and_target_get(dataset)
    print("\n=== Árvore de decisão ===")
    print("--- Treinamento e seleção de hiperparâmetros: Árvore de decisão ---")
    scores: list[float] = []
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
        best_params = search.best_params_
        score = float(search.score(features[test_indices], target[test_indices]))
        scores.append(score)
        print(
            f"Melhores hiperparâmetros da fold externa {fold}: "
            f"max_depth={best_params['max_depth']}, "
            f"min_samples_split={best_params['min_samples_split']}, "
            f"min_samples_leaf={best_params['min_samples_leaf']}; "
            f"R² médio interno={search.best_score_:.4f}; "
            f"R² externo={score:.4f}"
        )

    model_evaluation_print("Árvore de decisão", scores)

    return scores


def random_forest_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
    dataset = dataset_lazy.collect()
    features, target = features_and_target_get(dataset)
    print("\n=== Floresta aleatória ===")
    print("--- Treinamento e seleção de hiperparâmetros: Floresta aleatória ---")
    scores: list[float] = []
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
        best_params = search.best_params_
        score = float(search.score(features[test_indices], target[test_indices]))
        scores.append(score)
        print(
            f"Melhores hiperparâmetros da fold externa {fold}: "
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, "
            f"max_features={best_params['max_features']}; "
            f"R² médio interno={search.best_score_:.4f}; "
            f"R² externo={score:.4f}"
        )

    model_evaluation_print("Floresta aleatória", scores)

    return scores


def models_comparison_print(model_scores: list[tuple[str, list[float]]]) -> None:
    print("\n=== Comparação final ===")
    for model_name, scores in sorted(
        model_scores, key=lambda item: np.mean(item[1]), reverse=True
    ):
        print(
            f"{model_name}: R² médio={np.mean(scores):.4f}; "
            f"desvio-padrão={np.std(scores):.4f}"
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
    linear_regression_dataset_lazy = dataset.to_dummies(
        "ocean_proximity",
        drop_first=True,  # Use dummy variables for linear regression
    ).lazy()
    linear_regression_scores = linear_regression_train(linear_regression_dataset_lazy)

    tree_based_models_dataset_lazy = dataset.to_dummies(
        "ocean_proximity",
        drop_first=False,  # Use one-hot encoding for tree-based models
    ).lazy()
    decision_tree_scores = decision_tree_train(tree_based_models_dataset_lazy)
    random_forest_scores = random_forest_train(tree_based_models_dataset_lazy)
    models_comparison_print(
        [
            ("Regressão linear", linear_regression_scores),
            ("Árvore de decisão", decision_tree_scores),
            ("Floresta aleatória", random_forest_scores),
        ]
    )


if __name__ == "__main__":
    main()
