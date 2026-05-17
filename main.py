import polars.selectors as cs
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl


def main():
    DATASET_PATH = Path("dataset/housing.csv")
    OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING = (
        "ISLAND",
        "NEAR OCEAN",
        "<1H OCEAN",
        "NEAR BAY",
        "INLAND",
    )
    OCEAN_PROXIMITY_ENUM = pl.Enum(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING)
    data = pl.scan_csv(
        DATASET_PATH, schema_overrides={"ocean_proximity": OCEAN_PROXIMITY_ENUM}
    ).collect()

    print("Formato dos dados:", data.shape)

    DATASET_SAMPLE_LENGTH = 2000
    DATASET_SAMPLE_SEED = 3
    data = data.sample(
        DATASET_SAMPLE_LENGTH,
        with_replacement=False,
        shuffle=False,
        seed=DATASET_SAMPLE_SEED,
    )

    print("Formato da amostra:", data.shape)

    data = data.lazy()

    stats_descriptive = {
        "Quantidade de Observações": data.count(),
        "Quantidade de Valores Nulos": data.null_count(),
        "Média Aritmética": data.mean(),
        "1o quartil": data.quantile(0.25),
        "Mediana": data.median(),
        "3o quartil": data.quantile(0.75),
        "Desvio Padrão": data.std(ddof=1),  # ddof=1 para obter desvio padrão amostral
        "Amplitude": data.select(cs.numeric().max() - cs.numeric().min()),
    }

    STATS_DESCRIPTIVE_COLUMN_NAME = "Estatística"
    for stat_name, stat_lazyframe in stats_descriptive.items():
        stat_dataframe = stat_lazyframe.with_columns(
            pl.lit(stat_name).alias(STATS_DESCRIPTIVE_COLUMN_NAME)
        ).collect()

        print(stat_dataframe)

    PLOT_DIRECTORY_PATH = Path("plots/")
    # Empty the directory
    if PLOT_DIRECTORY_PATH.exists():
        for file in PLOT_DIRECTORY_PATH.iterdir():
            assert file.is_file()
            file.unlink()
    else:
        PLOT_DIRECTORY_PATH.mkdir()

    sns.set_theme()

    COLUMNS_EXCLUDED = ("longitude", "latitude")
    numeric_data = data.select(cs.numeric().exclude(COLUMNS_EXCLUDED)).collect()
    for column_name in numeric_data.columns:
        figure, (histplot_ax, boxplot_ax, ecdfplot_ax) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(10, 12),
            layout="constrained",
        )

        sns.histplot(data=numeric_data, x=column_name, kde=True, ax=histplot_ax)
        for container in histplot_ax.containers:
            histplot_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

        sns.boxplot(data=numeric_data, x=column_name, ax=boxplot_ax)

        sns.ecdfplot(data=numeric_data, x=column_name, ax=ecdfplot_ax)

        figure.savefig(
            fname=PLOT_DIRECTORY_PATH / f"{column_name}_histogram_&_boxplot_&_ecdf.png",
            bbox_inches="tight",
        )
        plt.close()

    categorical_data = data.select(
        cs.categorical().or_(cs.enum()).exclude(COLUMNS_EXCLUDED)
    ).collect()
    for column_name in categorical_data.columns:
        figure, (countplot_ax, piechart_ax) = plt.subplots(
            nrows=2, ncols=1, figsize=(10, 12), layout="constrained"
        )

        sns.countplot(
            data=categorical_data,
            x=column_name,
            ax=countplot_ax,
            order=OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING,
        )
        for container in countplot_ax.containers:
            countplot_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=16)
        counts_df = categorical_data[column_name].value_counts()
        counts = dict(counts_df.iter_rows())
        # Add "0" label to categories with no observations
        for i, category in enumerate(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING):
            if counts.get(category, 0) == 0:
                countplot_ax.annotate(
                    "0",
                    xy=(i, 0),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    clip_on=False,
                )

        # TODO: Add pie chart

        figure.savefig(
            fname=PLOT_DIRECTORY_PATH / f"{column_name}_bar_chart.png",
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    main()
