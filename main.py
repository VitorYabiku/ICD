import logging
from contextlib import contextmanager
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns

logger = logging.getLogger(__name__)
LOG_SPACING_VERTICAL_LINE_COUNT = 2


TABLE_DIRECTORY_PATH = Path("tables/")


def statistics_descriptive(data_lazyframe: pl.LazyFrame, filename_prefix: str):
    numeric_columns = cs.numeric()
    first_quartile = numeric_columns.quantile(0.25)
    third_quartile = numeric_columns.quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    outliers = (numeric_columns < first_quartile - 1.5 * interquartile_range) | (
        numeric_columns > third_quartile + 1.5 * interquartile_range
    )

    amplitude = numeric_columns.max() - numeric_columns.min()

    stats_descriptive = {
        "Quantidade de Observações": data_lazyframe.count(),
        "Quantidade de Valores Nulos": data_lazyframe.null_count(),
        "Média Aritmética": data_lazyframe.mean(),
        "1o quartil": data_lazyframe.quantile(0.25),
        "Mediana": data_lazyframe.median(),
        "3o quartil": data_lazyframe.quantile(0.75),
        "Desvio Padrão": data_lazyframe.std(
            ddof=1
        ),  # ddof=1 para obter desvio padrão amostral
        "Amplitude": data_lazyframe.select(amplitude),
        "Frequência Absoluta de Outliers": data_lazyframe.select(outliers.sum()),
        "Frequência Relativa de Outliers (%)": data_lazyframe.select(
            outliers.sum() * 100 / numeric_columns.count()
        ),
    }

    STATS_DESCRIPTIVE_COLUMN_NAME = "Estatística"
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
        / f"{filename_prefix}observacoes_com_algum_valor_null_ou_nan.json"
    )


PLOT_DIRECTORY_PATH = Path("plots/")


@contextmanager
def subplots(*args, savefig_path: Path, **kwargs):
    figure, axes = plt.subplots(*args, **kwargs)
    try:
        yield axes
    except Exception:
        raise
    else:
        figure.savefig(fname=savefig_path, bbox_inches="tight", dpi=300)
    finally:
        plt.close(figure)


def median_income_scatterplot_bivariate(
    data_lazyframe: pl.LazyFrame, column_other: str
):
    data = data_lazyframe.collect()
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


def data_numeric_plot(data_lazyframe: pl.LazyFrame):
    data = data_lazyframe.collect()
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
                t"Histograma com gráfico de densidade de {column_name}"
            )

            sns.boxplot(data=data, x=column_name, ax=boxplot_ax)
            boxplot_ax.set_title(t"Boxplot de {column_name}")

            sns.ecdfplot(data=data, x=column_name, ax=ecdfplot_ax)
            ecdfplot_ax.set_title(t"Gráfico de frequência acumulada de {column_name}")

    logger.info(
        f"data_numeric_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def correlation_matrix_plot(data_lazyframe: pl.LazyFrame):
    data = data_lazyframe.drop_nulls().collect()
    logger.info("EXECUTANDO correlation_matrix_plot com o seguinte dataframe...")
    logger.info("%s", data.head(1))

    correlation = data.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))

    with subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 12),
        layout="constrained",
        savefig_path=PLOT_DIRECTORY_PATH / "matrix_de_correlacao_de_pearson.png",
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


OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING = (
    "ISLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "<1H OCEAN",
    "INLAND",
)

COLUMNS_GEOSPATIAL = ("longitude", "latitude", "ocean_proximity")


def ocean_proximity_plot(data_lazyframe: pl.LazyFrame):
    data = data_lazyframe.collect()
    logger.info("EXECUTANDO ocean_proximity_plot com o seguinte dataframe...")
    logger.info("%s", data.head(1))

    column_name = "ocean_proximity"
    with subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
        layout="constrained",
        savefig_path=PLOT_DIRECTORY_PATH
        / f"{column_name}_grafico_de_barras_&_grafico_de_pizz.png",
    ) as (countplot_ax, piechart_ax):
        sns.countplot(
            data=data,
            x=column_name,
            ax=countplot_ax,
            order=OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING,
        )
        for container in countplot_ax.containers:
            countplot_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=16)
        counts_df = data[column_name].value_counts()
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
        countplot_ax.set_title(t"Gráfico de barras de {column_name}")

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
            autopct=lambda pct: t"{pct:.1f}%",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
            textprops={"fontsize": 12},
        )
        piechart_ax.set_title(t"Gráfico de pizza de {column_name}")
        piechart_ax.axis("equal")

    # geospatial analysis - begin
    geospatial_data = data_lazyframe.select(pl.col(COLUMNS_GEOSPATIAL)).collect()
    geospatial_pandas = geospatial_data.to_pandas()
    geospatial_geodataframe = gpd.GeoDataFrame(
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
            source=cx.providers.CartoDB.PositronNoLabels,
            attribution=False,
        )
        geospatial_ax.set_title("Distribuição geoespacial em relação a ocean_proximity")
        geospatial_ax.set_axis_off()
        geospatial_ax.legend(
            title="ocean_proximity",
            loc="lower left",
        )
    # geospatial analysis - end

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


def median_income_scatterplots(data_lazyframe: pl.LazyFrame):
    for column_name in data_lazyframe.collect_schema().names():
        if column_name != "median_income":
            median_income_scatterplot_bivariate(data_lazyframe, column_name)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(module)s.%(funcName)s:%(message)s"
    )

    DATASET_PATH = Path("dataset/housing.csv")
    OCEAN_PROXIMITY_ENUM = pl.Enum(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING)
    data = pl.scan_csv(
        DATASET_PATH, schema_overrides={"ocean_proximity": OCEAN_PROXIMITY_ENUM}
    ).collect()

    logger.info("Formato dos dados: %s", data.shape)
    logger.info(f"%s{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}", data.head())

    DATASET_SAMPLE_LENGTH = 2000
    DATASET_SAMPLE_SEED = 42
    data = data.sample(
        DATASET_SAMPLE_LENGTH,
        with_replacement=False,
        shuffle=False,
        seed=DATASET_SAMPLE_SEED,
    )

    logger.info("Formato da amostra: %s", data.shape)
    logger.info(f"%s{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}", data.head())

    data = data.lazy()

    # Empty the table directory
    logger.info("Resetando diretório de tabelas (%s)", TABLE_DIRECTORY_PATH)
    if TABLE_DIRECTORY_PATH.exists():
        for file in TABLE_DIRECTORY_PATH.iterdir():
            assert file.is_file()
            file.unlink()
    else:
        TABLE_DIRECTORY_PATH.mkdir()
    logger.info(
        f"Diretório de tabelas resetado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )

    statistics_descriptive(data, "")

    # Empty the plot directory
    logger.info("Resetando diretório de gráficos (%s)", PLOT_DIRECTORY_PATH)
    if PLOT_DIRECTORY_PATH.exists():
        for file in PLOT_DIRECTORY_PATH.iterdir():
            assert file.is_file()
            file.unlink()
    else:
        PLOT_DIRECTORY_PATH.mkdir()
    logger.info(
        f"Diretório de gráficos resetado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )

    sns.set_theme()

    data_numeric = data.select(cs.numeric().exclude(COLUMNS_GEOSPATIAL))
    data_numeric_plot(data_numeric)
    median_income_scatterplots(data_numeric)
    correlation_matrix_plot(data_numeric)

    OCEAN_PROXIMITY_COLUMNS_ADDITIONAL = "median_income"
    ocean_proximity_data = data.select(
        pl.col(*COLUMNS_GEOSPATIAL, OCEAN_PROXIMITY_COLUMNS_ADDITIONAL)
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
        pl.col(LOG_TRANSFORM_COLUMNS).log().name.suffix("_logaritmo_natural"),
        pl.col("median_income"),
    )
    data_numeric_plot(data_transformed_log)

    data = data.select(
        (pl.col("total_rooms") / pl.col("households")).alias("rooms_per_household"),
        (pl.col("total_bedrooms") / pl.col("total_rooms")).alias("bedrroms_per_room"),
        (pl.col("population") / pl.col("households")).alias("populaton_per_household"),
        pl.col("median_income"),
    )

    statistics_descriptive(data, "dados_transformados_")
    data_numeric_plot(data)
    median_income_scatterplots(data)


if __name__ == "__main__":
    main()
