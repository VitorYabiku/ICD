from pathlib import Path
from typing import Final

import polars as pl
import polars.selectors as cs

EP3_DIRECTORY_PATH: Final[Path] = Path(__file__).resolve().parent
PROJECT_DIRECTORY_PATH: Final[Path] = EP3_DIRECTORY_PATH.parent
DATASET_DIRECTORY_PATH: Final[Path] = PROJECT_DIRECTORY_PATH / "dataset"
DATASET_PATH: Final[Path] = DATASET_DIRECTORY_PATH / "housing.csv"

TARGET_COLUMN_NAME: Final = "median_income"
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
MIN_MAX_SCALE_EXPR: Final[pl.Expr] = (cs.numeric() - cs.numeric().min()) / (
    cs.numeric().max() - cs.numeric().min()
)
Z_SCORE_SCALE_EXPR: Final[pl.Expr] = (
    cs.numeric() - cs.numeric().mean()
) / cs.numeric().std()


def data_train_test_val_split(
    dataset_lazy: pl.LazyFrame,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    dataset_row_count: int = dataset_lazy.select(pl.len()).collect().item()

    TRAIN_FRACTION: Final[float] = 0.8
    TRAIN_ROW_COUNT: Final = int(TRAIN_FRACTION * dataset_row_count)
    data_train_lazy: pl.LazyFrame = dataset_lazy.head(TRAIN_ROW_COUNT).with_columns(
        MIN_MAX_SCALE_EXPR
    )

    TEST_FRACTION: Final[float] = 0.1
    TEST_ROW_COUNT: Final = int(TEST_FRACTION * dataset_row_count)
    data_test_lazy: pl.LazyFrame = dataset_lazy.slice(
        TRAIN_ROW_COUNT, TEST_ROW_COUNT
    ).with_columns(MIN_MAX_SCALE_EXPR)

    data_val_lazy: pl.LazyFrame = dataset_lazy.slice(
        TRAIN_ROW_COUNT + TEST_ROW_COUNT, None
    ).with_columns(MIN_MAX_SCALE_EXPR)

    return data_train_lazy, data_test_lazy, data_val_lazy


def linear_regression_train(dataset_lazy: pl.LazyFrame):
    dataset_lazy: pl.LazyFrame = (
        dataset_lazy.collect()
        .to_dummies(
            "ocean_proximity",
            drop_first=True,  # Dummy variables for linear regression
        )
        .lazy()
    )


def tree_based_models_train(dataset_lazy: pl.LazyFrame):
    dataset_lazy: pl.LazyFrame = (
        dataset_lazy.collect()
        .to_dummies(
            "ocean_proximity",
            drop_first=False,  # One-hot encoding for tree-based model
        )
        .lazy()
    )


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
    dataset_lazy = (
        dataset_lazy.filter(~ROW_WITH_NULL_OR_NAN_EXPR)
        .select(
            TARGET_COLUMN_NAME,
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
    ).lazy()

    data_train_lazy, data_test_lazy, data_val_lazy = data_train_test_val_split(
        dataset_lazy
    )

    print(data_train_lazy.collect())
    print(data_test_lazy.collect())
    print(data_val_lazy.collect())


if __name__ == "__main__":
    main()
