from pathlib import Path
from typing import Final

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


def data_train_test_val_split(
    dataset_lazy: pl.LazyFrame,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    dataset_row_count: int = dataset_lazy.select(pl.len()).collect().item()

    TRAIN_FRACTION: Final[float] = 0.8
    TRAIN_ROW_COUNT: Final = int(TRAIN_FRACTION * dataset_row_count)
    data_train_lazy: pl.LazyFrame = dataset_lazy.head(TRAIN_ROW_COUNT)

    TEST_FRACTION: Final[float] = 0.1
    TEST_ROW_COUNT: Final = int(TEST_FRACTION * dataset_row_count)
    data_test_lazy: pl.LazyFrame = dataset_lazy.slice(TRAIN_ROW_COUNT, TEST_ROW_COUNT)

    data_val_lazy: pl.LazyFrame = dataset_lazy.slice(
        TRAIN_ROW_COUNT + TEST_ROW_COUNT, None
    )

    return data_train_lazy, data_test_lazy, data_val_lazy


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
    train_lazy: pl.LazyFrame,
    test_lazy: pl.LazyFrame,
    val_lazy: pl.LazyFrame,
):
    train_numeric_statistics_lazy = train_lazy.select(NUMERIC_STATISTICS_EXPRESSIONS)
    train_lazy = linear_regression_data_preprocess(
        train_lazy, train_numeric_statistics_lazy
    )
    test_lazy = linear_regression_data_preprocess(
        test_lazy, train_numeric_statistics_lazy
    )
    val_lazy = linear_regression_data_preprocess(
        val_lazy, train_numeric_statistics_lazy
    )

    train = train_lazy.collect()
    train_y: np.ndarray = train.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    train_X: np.ndarray = train.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    linear_regression_model: Final = LinearRegression().fit(train_X, train_y)

    test = test_lazy.collect()
    test_X: np.ndarray = test.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    test_y: np.ndarray = test.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    print(linear_regression_model.score(test_X, test_y))


def tree_based_models_train(
    train_lazy: pl.LazyFrame,
    test_lazy: pl.LazyFrame,
    val_lazy: pl.LazyFrame,
):
    train = train_lazy.collect()
    train_y: np.ndarray = train.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    train_X: np.ndarray = train.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()

    RANDOM_STATE: Final[int] = 42
    decision_tree_model: Final = DecisionTreeRegressor(random_state=RANDOM_STATE).fit(
        train_X, train_y
    )
    random_forest_model: Final = RandomForestRegressor(random_state=RANDOM_STATE).fit(
        train_X, train_y
    )

    test = test_lazy.collect()
    test_X: np.ndarray = test.drop(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    test_y: np.ndarray = test.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    print(f"Decision tree test R²: {decision_tree_model.score(test_X, test_y)}")
    print(f"Random forest test R²: {random_forest_model.score(test_X, test_y)}")


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
    linear_regression_dataset_lazy = dataset.to_dummies(
        "ocean_proximity",
        drop_first=True,  # Use dummy variables for linear regression
    ).lazy()

    data_train_lazy, data_test_lazy, data_val_lazy = data_train_test_val_split(
        linear_regression_dataset_lazy
    )

    print(data_train_lazy.collect())
    print(data_test_lazy.collect())
    print(data_val_lazy.collect())

    linear_regression_train(data_train_lazy, data_test_lazy, data_val_lazy)

    tree_based_models_dataset_lazy = dataset.to_dummies(
        "ocean_proximity",
        drop_first=False,  # Use one-hot encoding for tree-based models
    ).lazy()
    data_train_lazy, data_test_lazy, data_val_lazy = data_train_test_val_split(
        tree_based_models_dataset_lazy
    )
    tree_based_models_train(data_train_lazy, data_test_lazy, data_val_lazy)


if __name__ == "__main__":
    main()
