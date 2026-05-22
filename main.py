import logging
from pathlib import Path

import contextily as cx
import geopandas as gpd
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns

from utils import reset_directory, subplots

logger = logging.getLogger(__name__)
LOG_SPACING_VERTICAL_LINE_COUNT: int = 2


TABLE_DIRECTORY_PATH: Path = Path("tables/")


def statistics_descriptive(
    data_lazyframe: pl.LazyFrame, filename_prefix: str
) -> None:
    numeric_columns: cs.Selector = cs.numeric()
    first_quartile: pl.Expr = numeric_columns.quantile(0.25)
    third_quartile: pl.Expr = numeric_columns.quantile(0.75)
    interquartile_range: pl.Expr = third_quartile - first_quartile
    outliers: pl.Expr = (numeric_columns < first_quartile - 1.5 * interquartile_range) | (
        numeric_columns > third_quartile + 1.5 * interquartile_range
    )

    amplitude: pl.Expr = numeric_columns.max() - numeric_columns.min()

    stats_descriptive: dict[str, pl.LazyFrame] = {
        "Quantidade de Observações": data_lazyframe.count(),
        "Quantidade de Valores Nulos": data_lazyframe.null_count(),
        "Média Aritmética": data_lazyframe.mean(),
        "1º quartil": data_lazyframe.quantile(0.25),
        "Mediana": data_lazyframe.median(),
        "3º quartil": data_lazyframe.quantile(0.75),
        "Desvio Padrão": data_lazyframe.std(
            ddof=1
        ),  # ddof=1 para obter desvio padrão amostral
        "Amplitude": data_lazyframe.select(amplitude),
        "Frequência Absoluta de Outliers": data_lazyframe.select(outliers.sum()),
        "Frequência Relativa de Outliers (%)": data_lazyframe.select(
            outliers.sum() * 100 / numeric_columns.count()
        ),
    }

    STATS_DESCRIPTIVE_COLUMN_NAME: str = "Estatística"
    for stat_name, stat_lazyframe in stats_descriptive.items():
        stat_lazyframe.with_columns(
            pl.lit(stat_name).alias(STATS_DESCRIPTIVE_COLUMN_NAME)
        ).collect().write_json(
            TABLE_DIRECTORY_PATH
            / f"{filename_prefix}estatisticas_descritivas_{stat_name}.json"
        )

    data_lazyframe.filter(
        pl.any_horizontal(pl.all().is_null()) | pl.any_horizontal(cs.float().is_nan())
    ).with_columns(
        pl.lit(
            "Observações com algum valor nulo ou número de ponto flutuante NaN"
        ).alias(STATS_DESCRIPTIVE_COLUMN_NAME)
    ).collect().write_json(
        TABLE_DIRECTORY_PATH
        / f"{filename_prefix}observacoes_com_algum_nan_ou_valor_nulo.json"
    )


PLOT_DIRECTORY_PATH: Path = Path("plots/")


def median_income_scatterplot_bivariate(
    data_lazyframe: pl.LazyFrame, column_other: str
) -> None:
    data: pl.DataFrame = data_lazyframe.collect()
    logger.info(
        "EXECUTANDO median_income_scatterplot_bivariate com o seguinte dataframe..."
    )
    logger.info("%s", data.head(1))

    with subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 12),
        savefig_path=PLOT_DIRECTORY_PATH
        / f"{column_other}_&_median_income_grafico_de_espalhamento.png",
    ) as ax:
        sns.scatterplot(
            data=data,
            x="median_income",
            y=column_other,
            ax=ax,
        )

    logger.info(
        f"median_income_scatterplot_bivariate executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def data_numeric_plot(data_lazyframe: pl.LazyFrame) -> None:
    data: pl.DataFrame = data_lazyframe.collect()
    logger.info("EXECUTANDO data_numeric_plot com o seguinte dataframe...")
    logger.info("%s", data.head(1))

    for column_name in data.columns:
        with subplots(
            nrows=3,
            ncols=1,
            figsize=(10, 12),
            layout="constrained",
            savefig_path=PLOT_DIRECTORY_PATH
            / f"{column_name}_histograma_&_boxplot_&_grafico_de_frequencia_acumulada.png",
        ) as (histplot_ax, boxplot_ax, ecdfplot_ax):
            sns.histplot(
                data=data, x=column_name, kde=True, kde_kws={"cut": 0}, ax=histplot_ax
            )
            for container in histplot_ax.containers:
                histplot_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
            histplot_ax.set_title(
                f"Histograma com gráfico de densidade de {column_name}"
            )

            sns.boxplot(data=data, x=column_name, ax=boxplot_ax)
            boxplot_ax.set_title(f"Boxplot de {column_name}")

            sns.ecdfplot(data=data, x=column_name, ax=ecdfplot_ax)
            ecdfplot_ax.set_title(f"Gráfico de frequência acumulada de {column_name}")

    logger.info(
        f"data_numeric_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def correlation_matrix_plot(data_lazyframe: pl.LazyFrame) -> None:
    data: pl.DataFrame = data_lazyframe.drop_nulls().collect()
    logger.info("EXECUTANDO correlation_matrix_plot com o seguinte dataframe...")
    logger.info("%s", data.head(1))

    correlation: pl.DataFrame = data.corr()
    mask: np.ndarray = np.triu(np.ones_like(correlation, dtype=bool))

    with subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 12),
        layout="constrained",
        savefig_path=PLOT_DIRECTORY_PATH / "matriz_de_correlacao_de_pearson.png",
    ) as correlation_ax:
        sns.heatmap(
            correlation.to_numpy(),
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=correlation.columns,
            yticklabels=correlation.columns,
            ax=correlation_ax,
        )
        correlation_ax.set_title(
            "Matriz de correlação de Pearson das variáveis quantitativas"
        )

    logger.info(
        f"correlation_matrix_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING: list[str] = [
    "ISLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "<1H OCEAN",
    "INLAND",
]

COLUMNS_GEOSPATIAL: list[str] = ["longitude", "latitude", "ocean_proximity"]


def ocean_proximity_plot(data_lazyframe: pl.LazyFrame) -> None:
    data: pl.DataFrame = data_lazyframe.collect()
    logger.info("EXECUTANDO ocean_proximity_plot com o seguinte dataframe...")
    logger.info("%s", data.head(1))

    column_name: str = "ocean_proximity"
    with subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
        layout="constrained",
        savefig_path=PLOT_DIRECTORY_PATH
        / f"{column_name}_grafico_de_barras_&_grafico_de_pizza.png",
    ) as (countplot_ax, piechart_ax):
        sns.countplot(
            data=data,
            x=column_name,
            ax=countplot_ax,
            order=OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING,
        )
        for container in countplot_ax.containers:
            countplot_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=16)
        counts_df: pl.DataFrame = data[column_name].value_counts()
        counts: dict[str, int] = dict(counts_df.iter_rows())
        counts_ordered: list[int] = [
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

        labels_nonzero: list[str] = [
            category
            for category, count in zip(
                OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING, counts_ordered
            )
            if count > 0
        ]
        counts_nonzero: list[int] = [count for count in counts_ordered if count > 0]
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

    # Geospatial analysis - begin
    geospatial_data: pl.DataFrame = data_lazyframe.select(
        pl.col(COLUMNS_GEOSPATIAL)
    ).collect()
    geospatial_pandas = geospatial_data.to_pandas()
    geospatial_geodataframe: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geospatial_pandas,
        geometry=gpd.points_from_xy(
            geospatial_pandas["longitude"], geospatial_pandas["latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    with subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 12),
        layout="constrained",
        savefig_path=PLOT_DIRECTORY_PATH / "ocean_proximity_mapa_geoespacial.png",
    ) as geospatial_ax:
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
            # False positive: xyzservices providers are dynamic Bunch attributes;
            # runtime `cx.providers.CartoDB.PositronNoLabels` exists.
            source=cx.providers.CartoDB.PositronNoLabels,  # pyrefly: ignore[missing-attribute]
            attribution=False,
        )
        geospatial_ax.set_title("Distribuição geoespacial em relação a ocean_proximity")
        geospatial_ax.set_axis_off()
        geospatial_ax.legend(
            title="ocean_proximity",
            loc="lower left",
        )
    # Geospatial analysis - end

    with subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 12),
        savefig_path=PLOT_DIRECTORY_PATH
        / "median_income_por_ocean_proximity_boxplot.png",
    ) as boxplot_ax:
        sns.boxplot(
            data=data,
            x="ocean_proximity",
            y="median_income",
            order=OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING,
            ax=boxplot_ax,
        )

        boxplot_ax.set_title("Boxplot de median_income por ocean_proximity")

    logger.info(
        f"ocean_proximity_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def median_income_scatterplots(data_lazyframe: pl.LazyFrame) -> None:
    for column_name in data_lazyframe.collect_schema().names():
        if column_name != "median_income":
            median_income_scatterplot_bivariate(data_lazyframe, column_name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(module)s.%(funcName)s:%(message)s"
    )

    DATASET_PATH: Path = Path("dataset/housing.csv")
    OCEAN_PROXIMITY_ENUM: pl.Enum = pl.Enum(
        OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING
    )
    data: pl.DataFrame = pl.scan_csv(
        DATASET_PATH, schema_overrides={"ocean_proximity": OCEAN_PROXIMITY_ENUM}
    ).collect()

    logger.info("Formato dos dados: %s", data.shape)
    logger.info(f"%s{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}", data.head())

    DATASET_SAMPLE_LENGTH: int = 2064
    DATASET_SAMPLE_SEED: int = 42
    data = data.sample(
        DATASET_SAMPLE_LENGTH,
        with_replacement=False,
        shuffle=False,
        seed=DATASET_SAMPLE_SEED,
    )

    logger.info("Formato da amostra: %s", data.shape)
    logger.info(f"%s{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}", data.head())

    data_lazy: pl.LazyFrame = data.lazy()

    reset_directory(TABLE_DIRECTORY_PATH, "tabelas")

    statistics_descriptive(data_lazy, "")

    reset_directory(PLOT_DIRECTORY_PATH, "gráficos")

    sns.set_theme()

    data_numeric: pl.LazyFrame = data_lazy.select(
        cs.numeric().exclude(COLUMNS_GEOSPATIAL)
    )
    data_numeric_plot(data_numeric)
    median_income_scatterplots(data_numeric)
    correlation_matrix_plot(data_numeric)

    OCEAN_PROXIMITY_COLUMNS_ADDITIONAL: list[str] = ["median_income"]
    ocean_proximity_data: pl.LazyFrame = data_lazy.select(
        pl.col(*COLUMNS_GEOSPATIAL, *OCEAN_PROXIMITY_COLUMNS_ADDITIONAL)
    )
    ocean_proximity_plot(ocean_proximity_data)

    LOG_TRANSFORM_COLUMNS: list[str] = [
        "households",
        "median_income",
        "population",
        "total_rooms",
        "total_bedrooms",
    ]
    data_transformed_log: pl.LazyFrame = data_lazy.select(
        pl.col(LOG_TRANSFORM_COLUMNS).log().name.suffix("_logaritmo_natural"),
        pl.col("median_income"),
    )
    data_numeric_plot(data_transformed_log)

    DATA_WITH_VARIABLES_NEW_COLUMNS: list[str] = [
        "total_rooms",
        "households",
        "total_bedrooms",
        "population",
        "median_income",
    ]
    data_with_variables_new: pl.LazyFrame = data_lazy.select(
        (pl.col("total_rooms") / pl.col("households")).alias("rooms_per_household"),
        (pl.col("total_bedrooms") / pl.col("total_rooms")).alias("bedrooms_per_room"),
        (pl.col("population") / pl.col("households")).alias("population_per_household"),
        pl.col(*DATA_WITH_VARIABLES_NEW_COLUMNS),
    )

    statistics_descriptive(
        data_with_variables_new, filename_prefix="dados_transformados_"
    )
    data_numeric_plot(data_with_variables_new)
    median_income_scatterplots(data_with_variables_new)
    correlation_matrix_plot(data_with_variables_new)

    # Extra analysis - begin

    # Análise de outliers de population_per_household
    data_with_variables_new.sort(
        pl.col("population_per_household"), descending=True, nulls_last=True
    ).head(20).collect().write_json(
        TABLE_DIRECTORY_PATH / "population_per_household_outliers.json"
    )
    # Extra analysis - end


if __name__ == "__main__":
    main()
