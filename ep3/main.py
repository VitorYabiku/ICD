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
Z_SCORE_SCALE_EXPR: Final[pl.Expr] = (
    cs.numeric() - cs.numeric().mean()
) / cs.numeric().std()

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


def linear_regression_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
    scores: list[float] = []

    for test_fold in range(FOLD_COUNT):
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
        train_y: np.ndarray = train.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        train_X: np.ndarray = train.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        linear_regression_model: Final = LinearRegression().fit(train_X, train_y)

        test = test_lazy.collect()
        test_X: np.ndarray = test.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        test_y: np.ndarray = test.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        score = float(linear_regression_model.score(test_X, test_y))
        scores.append(score)
        print(f"Linear regression fold {test_fold + 1} test R²: {score}")

    print(f"Linear regression mean test R²: {np.mean(scores)}")
    print(f"Linear regression test R² standard deviation: {np.std(scores)}")

    return scores


def decision_tree_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
    scores: list[float] = []

    for test_fold in range(FOLD_COUNT):
        validation_fold = (test_fold + 1) % FOLD_COUNT

        train_lazy = dataset_lazy.filter(
            ~pl.col("fold").is_in([test_fold, validation_fold])
        ).drop("fold")
        test_lazy = dataset_lazy.filter(pl.col("fold") == test_fold).drop("fold")
        validation_lazy = dataset_lazy.filter(
            pl.col("fold") == validation_fold
        ).drop("fold")
        final_train_lazy = dataset_lazy.filter(pl.col("fold") != test_fold).drop(
            "fold"
        )

        train = train_lazy.collect()
        train_y: np.ndarray = train.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        train_X: np.ndarray = train.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()

        validation = validation_lazy.collect()
        validation_X: np.ndarray = validation.drop(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()
        validation_y: np.ndarray = validation.get_column(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()

        best_validation_score = -np.inf
        best_max_depth: int | None = None
        best_min_samples_split = 2
        best_min_samples_leaf = 1
        for max_depth in DECISION_TREE_MAX_DEPTH_VALUES:
            for min_samples_split in DECISION_TREE_MIN_SAMPLES_SPLIT_VALUES:
                for min_samples_leaf in DECISION_TREE_MIN_SAMPLES_LEAF_VALUES:
                    candidate_model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=TREE_BASED_MODELS_RANDOM_STATE,
                    ).fit(train_X, train_y)
                    validation_score = candidate_model.score(
                        validation_X, validation_y
                    )
                    if validation_score > best_validation_score:
                        best_validation_score = validation_score
                        best_max_depth = max_depth
                        best_min_samples_split = min_samples_split
                        best_min_samples_leaf = min_samples_leaf

        print(
            f"Decision tree fold {test_fold + 1} best hyperparameters: "
            f"max_depth={best_max_depth}, "
            f"min_samples_split={best_min_samples_split}, "
            f"min_samples_leaf={best_min_samples_leaf}; "
            f"validation R²: {best_validation_score}"
        )

        final_train = final_train_lazy.collect()
        final_train_y: np.ndarray = final_train.get_column(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()
        final_train_X: np.ndarray = final_train.drop(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()
        decision_tree_model: Final = DecisionTreeRegressor(
            max_depth=best_max_depth,
            min_samples_split=best_min_samples_split,
            min_samples_leaf=best_min_samples_leaf,
            random_state=TREE_BASED_MODELS_RANDOM_STATE,
        ).fit(final_train_X, final_train_y)

        test = test_lazy.collect()
        test_X: np.ndarray = test.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        test_y: np.ndarray = test.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        score = float(decision_tree_model.score(test_X, test_y))
        scores.append(score)
        print(f"Decision tree fold {test_fold + 1} test R²: {score}")

    print(f"Decision tree mean test R²: {np.mean(scores)}")
    print(f"Decision tree test R² standard deviation: {np.std(scores)}")

    return scores


def random_forest_train(
    dataset_lazy: pl.LazyFrame,
) -> list[float]:
    scores: list[float] = []

    for test_fold in range(FOLD_COUNT):
        validation_fold = (test_fold + 1) % FOLD_COUNT

        train_lazy = dataset_lazy.filter(
            ~pl.col("fold").is_in([test_fold, validation_fold])
        ).drop("fold")
        test_lazy = dataset_lazy.filter(pl.col("fold") == test_fold).drop("fold")
        validation_lazy = dataset_lazy.filter(
            pl.col("fold") == validation_fold
        ).drop("fold")
        final_train_lazy = dataset_lazy.filter(pl.col("fold") != test_fold).drop(
            "fold"
        )

        train = train_lazy.collect()
        train_y: np.ndarray = train.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        train_X: np.ndarray = train.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()

        validation = validation_lazy.collect()
        validation_X: np.ndarray = validation.drop(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()
        validation_y: np.ndarray = validation.get_column(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()

        best_validation_score = -np.inf
        best_n_estimators = 50
        best_max_depth: int | None = None
        best_max_features: float | Literal["sqrt", "log2"] = 1.0
        for n_estimators in RANDOM_FOREST_N_ESTIMATORS_VALUES:
            for max_depth in RANDOM_FOREST_MAX_DEPTH_VALUES:
                for max_features in RANDOM_FOREST_MAX_FEATURES_VALUES:
                    candidate_model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        max_features=max_features,
                        random_state=TREE_BASED_MODELS_RANDOM_STATE,
                        n_jobs=-1,
                    ).fit(train_X, train_y)
                    validation_score = candidate_model.score(
                        validation_X, validation_y
                    )
                    if validation_score > best_validation_score:
                        best_validation_score = validation_score
                        best_n_estimators = n_estimators
                        best_max_depth = max_depth
                        best_max_features = max_features

        print(
            f"Random forest fold {test_fold + 1} best hyperparameters: "
            f"n_estimators={best_n_estimators}, "
            f"max_depth={best_max_depth}, "
            f"max_features={best_max_features}; "
            f"validation R²: {best_validation_score}"
        )

        final_train = final_train_lazy.collect()
        final_train_y: np.ndarray = final_train.get_column(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()
        final_train_X: np.ndarray = final_train.drop(
            TARGET_VARIABLE_COLUMN_NAME
        ).to_numpy()
        random_forest_model: Final = RandomForestRegressor(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            max_features=best_max_features,
            random_state=TREE_BASED_MODELS_RANDOM_STATE,
            n_jobs=-1,
        ).fit(final_train_X, final_train_y)

        test = test_lazy.collect()
        test_X: np.ndarray = test.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        test_y: np.ndarray = test.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        score = float(random_forest_model.score(test_X, test_y))
        scores.append(score)
        print(f"Random forest fold {test_fold + 1} test R²: {score}")

    print(f"Random forest mean test R²: {np.mean(scores)}")
    print(f"Random forest test R² standard deviation: {np.std(scores)}")

    return scores


def tree_based_models_train(
    dataset_lazy: pl.LazyFrame,
) -> None:
    decision_tree_train(dataset_lazy)
    random_forest_train(dataset_lazy)


def main():
    dataset_lazy: pl.LazyFrame = pl.scan_csv(
        DATASET_PATH,
        schema_overrides={
            "ocean_proximity": pl.Enum(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING)
        },
    )

    dataset_rows_with_null_or_nan = dataset_lazy.filter(ROW_WITH_NULL_OR_NAN_EXPR)
    print(
        f"Quantidade de linhas com algum null ou NaN: {
            dataset_rows_with_null_or_nan.select(pl.len()).collect().item()
        }"
    )

    DATA_SAMPLE_SEED: Final[int] = 42
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
    linear_regression_train(linear_regression_dataset_lazy)

    tree_based_models_dataset_lazy = data_folds_add(
        dataset.to_dummies(
            "ocean_proximity",
            drop_first=False,  # Use one-hot encoding for tree-based models
        ).lazy(),
        FOLD_COUNT,
    )
    tree_based_models_train(tree_based_models_dataset_lazy)


if __name__ == "__main__":
    main()
