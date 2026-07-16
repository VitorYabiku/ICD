from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

EP3_DIRECTORY_PATH: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIRECTORY_PATH: Final[Path] = EP3_DIRECTORY_PATH.parent
DATASET_DIRECTORY_PATH: Final[Path] = PROJECT_DIRECTORY_PATH / "dataset"
DATASET_PATH: Final[Path] = DATASET_DIRECTORY_PATH / "housing.csv"

FOLD_COUNT: Final[int] = 5
DATA_SAMPLE_SEED: Final[int] = 42
TREE_BASED_MODELS_RANDOM_STATE: Final[int] = 42
DECISION_TREE_MAX_DEPTH_VALUES: Final = (None, 3, 5, 10, 20)
DECISION_TREE_MIN_SAMPLES_SPLIT_VALUES: Final = (2, 5, 10)
DECISION_TREE_MIN_SAMPLES_LEAF_VALUES: Final = (1, 2, 4)
RANDOM_FOREST_N_ESTIMATORS_VALUES: Final = (50, 100, 200)
RANDOM_FOREST_MAX_DEPTH_VALUES: Final = (None, 5, 10, 20)
RANDOM_FOREST_MAX_FEATURES_VALUES: Final = (1.0, "sqrt", "log2")

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

ROW_WITH_NULL_OR_NAN_EXPR: Final[pl.Expr] = pl.any_horizontal(
    pl.all().is_null()
) | pl.any_horizontal(cs.float().is_nan())
NUMERIC_STATISTICS_COLUMN_NAMES: Final = [
    f"{column_name}_{statistic}"
    for column_name in NUMERIC_FEATURE_COLUMN_NAMES
    for statistic in ("min", "max")
]
NUMERIC_STATISTICS_EXPRESSIONS: Final = [
    expression
    for column_name in NUMERIC_FEATURE_COLUMN_NAMES
    for expression in (
        pl.col(column_name).min().alias(f"{column_name}_min"),
        pl.col(column_name).max().alias(f"{column_name}_max"),
    )
]


def data_folds_add(dataset_lazy: pl.LazyFrame, fold_count: int) -> pl.LazyFrame:
    return (
        dataset_lazy.with_row_index(name="index")
        .with_columns(
            (pl.col("index") % fold_count).alias("fold"),
        )
        .drop("index")
    )


def linear_regression_data_preprocess(
    dataset_lazy: pl.LazyFrame,
    train_numeric_statistics_lazy: pl.LazyFrame,
) -> pl.LazyFrame:
    min_max_scale_exprs = [
        pl.when(
            pl.col(min_column_name := f"{column_name}_min")
            == pl.col(max_column_name := f"{column_name}_max")
        )
        .then(0.0)
        .otherwise(
            (pl.col(column_name) - pl.col(min_column_name))
            / (pl.col(max_column_name) - pl.col(min_column_name))
        )
        .alias(column_name)
        for column_name in NUMERIC_FEATURE_COLUMN_NAMES
    ]

    return (
        dataset_lazy.join(train_numeric_statistics_lazy, how="cross")
        .with_columns(min_max_scale_exprs)
        .drop(NUMERIC_STATISTICS_COLUMN_NAMES)
    )


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
    scores: list[float] = []
    print("\n=== Regressão linear ===")
    print("--- Treinamento ---")

    for test_fold in range(FOLD_COUNT):
        print(f"Processando fold externa {test_fold + 1}/{FOLD_COUNT}...")
        train_lazy = dataset_lazy.filter(pl.col("fold") != test_fold).drop("fold")
        test_lazy = dataset_lazy.filter(pl.col("fold") == test_fold).drop("fold")
        train_numeric_statistics_lazy = train_lazy.select(
            NUMERIC_STATISTICS_EXPRESSIONS
        )
        train_lazy = linear_regression_data_preprocess(
            train_lazy, train_numeric_statistics_lazy
        )
        test_lazy = linear_regression_data_preprocess(
            test_lazy, train_numeric_statistics_lazy
        )

        train = train_lazy.collect()
        train_X, train_y = features_and_target_get(train)
        linear_regression_model: Final = LinearRegression().fit(train_X, train_y)

        test = test_lazy.collect()
        test_X, test_y = features_and_target_get(test)
        score = float(linear_regression_model.score(test_X, test_y))
        scores.append(score)

    model_evaluation_print("Regressão linear", scores)

    return scores


def decision_tree_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
    scores: list[float] = []
    dataset = dataset_lazy.collect()
    print("\n=== Árvore de decisão ===")
    print("--- Treinamento e seleção de hiperparâmetros ---")

    for test_fold in range(FOLD_COUNT):
        print(f"Processando fold externa {test_fold + 1}/{FOLD_COUNT}...")
        inner_folds = [fold for fold in range(FOLD_COUNT) if fold != test_fold]
        inner_splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for validation_fold in inner_folds:
            train = dataset.filter(
                ~pl.col("fold").is_in([test_fold, validation_fold])
            ).drop("fold")
            validation = dataset.filter(pl.col("fold") == validation_fold).drop("fold")
            train_X, train_y = features_and_target_get(train)
            validation_X, validation_y = features_and_target_get(validation)
            inner_splits.append((train_X, train_y, validation_X, validation_y))

        best_mean_validation_score = -np.inf
        best_max_depth: int | None = None
        best_min_samples_split = 2
        best_min_samples_leaf = 1
        for max_depth in DECISION_TREE_MAX_DEPTH_VALUES:
            for min_samples_split in DECISION_TREE_MIN_SAMPLES_SPLIT_VALUES:
                for min_samples_leaf in DECISION_TREE_MIN_SAMPLES_LEAF_VALUES:
                    validation_scores: list[float] = []
                    for train_X, train_y, validation_X, validation_y in inner_splits:
                        candidate_model = DecisionTreeRegressor(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=TREE_BASED_MODELS_RANDOM_STATE,
                        ).fit(train_X, train_y)
                        validation_scores.append(
                            float(candidate_model.score(validation_X, validation_y))
                        )
                    mean_validation_score = float(np.mean(validation_scores))
                    if mean_validation_score > best_mean_validation_score:
                        best_mean_validation_score = mean_validation_score
                        best_max_depth = max_depth
                        best_min_samples_split = min_samples_split
                        best_min_samples_leaf = min_samples_leaf

        print(
            f"Melhores hiperparâmetros da fold {test_fold + 1}: "
            f"max_depth={best_max_depth}, "
            f"min_samples_split={best_min_samples_split}, "
            f"min_samples_leaf={best_min_samples_leaf}; "
            f"R² médio interno={best_mean_validation_score:.4f}"
        )

        final_train = dataset.filter(pl.col("fold") != test_fold).drop("fold")
        final_train_X, final_train_y = features_and_target_get(final_train)
        decision_tree_model: Final = DecisionTreeRegressor(
            max_depth=best_max_depth,
            min_samples_split=best_min_samples_split,
            min_samples_leaf=best_min_samples_leaf,
            random_state=TREE_BASED_MODELS_RANDOM_STATE,
        ).fit(final_train_X, final_train_y)

        test = dataset.filter(pl.col("fold") == test_fold).drop("fold")
        test_X, test_y = features_and_target_get(test)
        scores.append(float(decision_tree_model.score(test_X, test_y)))

    model_evaluation_print("Árvore de decisão", scores)

    return scores


def random_forest_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
    scores: list[float] = []
    dataset = dataset_lazy.collect()
    print("\n=== Floresta aleatória ===")
    print("--- Treinamento e seleção de hiperparâmetros ---")

    for test_fold in range(FOLD_COUNT):
        print(f"Processando fold externa {test_fold + 1}/{FOLD_COUNT}...")
        inner_folds = [fold for fold in range(FOLD_COUNT) if fold != test_fold]
        inner_splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for validation_fold in inner_folds:
            train = dataset.filter(
                ~pl.col("fold").is_in([test_fold, validation_fold])
            ).drop("fold")
            validation = dataset.filter(pl.col("fold") == validation_fold).drop("fold")
            train_X, train_y = features_and_target_get(train)
            validation_X, validation_y = features_and_target_get(validation)
            inner_splits.append((train_X, train_y, validation_X, validation_y))

        best_mean_validation_score = -np.inf
        best_n_estimators = 50
        best_max_depth: int | None = None
        best_max_features: float | Literal["sqrt", "log2"] = 1.0
        for n_estimators in RANDOM_FOREST_N_ESTIMATORS_VALUES:
            for max_depth in RANDOM_FOREST_MAX_DEPTH_VALUES:
                for max_features in RANDOM_FOREST_MAX_FEATURES_VALUES:
                    validation_scores: list[float] = []
                    for train_X, train_y, validation_X, validation_y in inner_splits:
                        candidate_model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            max_features=max_features,
                            random_state=TREE_BASED_MODELS_RANDOM_STATE,
                            n_jobs=-1,
                        ).fit(train_X, train_y)
                        validation_scores.append(
                            float(candidate_model.score(validation_X, validation_y))
                        )
                    mean_validation_score = float(np.mean(validation_scores))
                    if mean_validation_score > best_mean_validation_score:
                        best_mean_validation_score = mean_validation_score
                        best_n_estimators = n_estimators
                        best_max_depth = max_depth
                        best_max_features = max_features

        print(
            f"Melhores hiperparâmetros da fold {test_fold + 1}: "
            f"n_estimators={best_n_estimators}, "
            f"max_depth={best_max_depth}, "
            f"max_features={best_max_features}; "
            f"R² médio interno={best_mean_validation_score:.4f}"
        )

        final_train = dataset.filter(pl.col("fold") != test_fold).drop("fold")
        final_train_X, final_train_y = features_and_target_get(final_train)
        random_forest_model: Final = RandomForestRegressor(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            max_features=best_max_features,
            random_state=TREE_BASED_MODELS_RANDOM_STATE,
            n_jobs=-1,
        ).fit(final_train_X, final_train_y)

        test = dataset.filter(pl.col("fold") == test_fold).drop("fold")
        test_X, test_y = features_and_target_get(test)
        scores.append(float(random_forest_model.score(test_X, test_y)))

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
        # Shuffle data before splitting
        .sample(fraction=1.0, shuffle=True, seed=DATA_SAMPLE_SEED)
    )
    linear_regression_dataset_lazy = data_folds_add(
        dataset.to_dummies(
            "ocean_proximity",
            drop_first=True,  # Use dummy variables for linear regression
        ).lazy(),
        FOLD_COUNT,
    )
    linear_regression_scores = linear_regression_train(linear_regression_dataset_lazy)

    tree_based_models_dataset_lazy = data_folds_add(
        dataset.to_dummies(
            "ocean_proximity",
            drop_first=False,  # Use one-hot encoding for tree-based models
        ).lazy(),
        FOLD_COUNT,
    )
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
