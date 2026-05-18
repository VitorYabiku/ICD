from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns


def statistics_descriptive(data: pl.LazyFrame):
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


def data_numeric_plot(data_lazyframe: pl.LazyFrame):
    data = data_lazyframe.collect()
    for column_name in data.columns:
        figure, (histplot_ax, boxplot_ax, ecdfplot_ax) = plt.subplots(
            nrows=3, ncols=1, figsize=(10, 12), layout="constrained"
        )

        sns.histplot(
            data=data, x=column_name, kde=True, kde_kws={"cut": 0}, ax=histplot_ax
        )
        for container in histplot_ax.containers:
            histplot_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
        histplot_ax.set_title(f"Histograma com gráfico de densidade de {column_name}")

        sns.boxplot(data=data, x=column_name, ax=boxplot_ax)
        boxplot_ax.set_title(f"Boxplot de {column_name}")

        sns.ecdfplot(data=data, x=column_name, ax=ecdfplot_ax)
        ecdfplot_ax.set_title(f"Gráfico de frequência acumulada de {column_name}")

        figure.savefig(
            fname=PLOT_DIRECTORY_PATH / f"{column_name}_histogram_&_boxplot_&_ecdf.png",
            bbox_inches="tight",
        )
        plt.close()


OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING = (
    "ISLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "<1H OCEAN",
    "INLAND",
)

COLUMNS_GEOSPATIAL = ("longitude", "latitude", "ocean_proximity")


def ocean_proximity_plot(data: pl.LazyFrame):
    data_collect = data.collect()

    figure, (countplot_ax, piechart_ax) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 12), layout="constrained"
    )
    column_name = "ocean_proximity"

    sns.countplot(
        data=data_collect,
        x=column_name,
        ax=countplot_ax,
        order=OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING,
    )
    for container in countplot_ax.containers:
        countplot_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=16)
    counts_df = data_collect[column_name].value_counts()
    counts = dict(counts_df.iter_rows())
    counts_ordered = [
        counts.get(category, 0)
        for category in OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING
    ]
    # Add "0" label to categories with no observations
    for i, category in enumerate(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING):
        if counts_ordered[i] == 0:
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
    countplot_ax.set_title(f"Gráfico de barras de {column_name}")

    labels_nonzero = [
        category
        for category, count in zip(
            OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING, counts_ordered
        )
        if count > 0
    ]
    counts_nonzero = [count for count in counts_ordered if count > 0]
    piechart_ax.pie(
        counts_nonzero,
        labels=labels_nonzero,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        textprops={"fontsize": 12},
    )
    piechart_ax.set_title(f"Gráfico de pizza de {column_name}")
    piechart_ax.axis("equal")

    figure.savefig(
        fname=PLOT_DIRECTORY_PATH / f"{column_name}_bar_chart_&_pie_chart.png",
        bbox_inches="tight",
    )
    plt.close()

    # geospatial analysis - begin
    geospatial_data = data.select(pl.col(COLUMNS_GEOSPATIAL)).collect()
    geospatial_pandas = geospatial_data.to_pandas()
    geospatial_geodataframe = gpd.GeoDataFrame(
        geospatial_pandas,
        geometry=gpd.points_from_xy(
            geospatial_pandas["longitude"], geospatial_pandas["latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    figure, geospatial_ax = plt.subplots(
        nrows=1, ncols=1, figsize=(10, 12), layout="constrained"
    )
    for ocean_proximity in OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING:
        geospatial_category = geospatial_geodataframe[
            geospatial_geodataframe["ocean_proximity"] == ocean_proximity
        ]
        if geospatial_category.empty:
            continue
        geospatial_category.plot(
            ax=geospatial_ax,
            markersize=18,
            alpha=0.65,
            label=ocean_proximity,
        )

    cx.add_basemap(
        geospatial_ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        attribution=False,
    )
    geospatial_ax.set_title("Distribuição geoespacial em relação a ocean_proximity")
    geospatial_ax.set_axis_off()
    geospatial_ax.legend(
        title="ocean_proximity",
        loc="lower left",
    )

    figure.savefig(
        fname=PLOT_DIRECTORY_PATH / "ocean_proximity_geospatial_map.png",
        bbox_inches="tight",
    )
    plt.close()
    # geospatial analysis - end

    figure, boxplot_ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 12))

    sns.boxplot(
        data=data_collect,
        x="ocean_proximity",
        y="median_income",
        order=OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING,
        ax=boxplot_ax,
    )

    boxplot_ax.set_title("Boxplot de median_income por ocean_proximity")

    figure.savefig(
        fname=PLOT_DIRECTORY_PATH / "median_income_by_ocean_proximity_boxplot.png",
        bbox_inches="tight",
    )


def main():
    DATASET_PATH = Path("dataset/housing.csv")
    OCEAN_PROXIMITY_ENUM = pl.Enum(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING)
    data = pl.scan_csv(
        DATASET_PATH, schema_overrides={"ocean_proximity": OCEAN_PROXIMITY_ENUM}
    ).collect()

    print("Formato dos dados:", data.shape)

    DATASET_SAMPLE_LENGTH = 2000
    DATASET_SAMPLE_SEED = 42
    data = data.sample(
        DATASET_SAMPLE_LENGTH,
        with_replacement=False,
        shuffle=False,
        seed=DATASET_SAMPLE_SEED,
    )

    print("Formato da amostra:", data.shape)

    data = data.lazy()

    statistics_descriptive(data)

    # Empty the directory
    if PLOT_DIRECTORY_PATH.exists():
        for file in PLOT_DIRECTORY_PATH.iterdir():
            assert file.is_file()
            file.unlink()
    else:
        PLOT_DIRECTORY_PATH.mkdir()

    sns.set_theme()

    data_numeric = data.select(cs.numeric().exclude(COLUMNS_GEOSPATIAL))
    data_numeric_plot(data_numeric)

    OCEAN_PROXIMITY_COLUMNS_ADDITIONAL = "median_income"
    ocean_proximity_data = data.select(
        pl.col(COLUMNS_GEOSPATIAL), pl.col(OCEAN_PROXIMITY_COLUMNS_ADDITIONAL)
    )
    ocean_proximity_plot(ocean_proximity_data)

    LOG_TRANSFORM_COLUMNS = (
        "households",
        "median_income",
        "population",
        "total_rooms",
        "total_bedrooms",
    )
    data_transformed_log = data.select(
        pl.col(LOG_TRANSFORM_COLUMNS).log().name.suffix("_logaritmo_natural")
    )
    data_numeric_plot(data_transformed_log)


if __name__ == "__main__":
    main()
