import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score

from utils import reset_directory, subplots

logger = logging.getLogger(__name__)
LOG_SPACING_VERTICAL_LINE_COUNT: int = 2


DATASET_PATH: Path = Path("dataset/housing.csv")
CLUSTERING_OUTPUT_DIRECTORY_PATH: Path = Path("clustering_outputs/")
HOUSING_STRATIFIED_PATH: Path = Path("housing_stratified.csv")
MEDIAN_INCOME_COLUMN_NAME: str = "median_income"
MEDIAN_INCOME_CLUSTER_COLUMN_NAME: str = "median_income_cluster"
LINKAGE_METHODS: list[str] = ["single", "complete", "average", "ward"]
CLUSTER_COUNTS: range = range(2, 11)
OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING: list[str] = [
    "ISLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "<1H OCEAN",
    "INLAND",
]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]


def data_complete_load() -> pl.DataFrame:
    ocean_proximity_enum: pl.Enum = pl.Enum(
        OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING
    )
    data: pl.DataFrame = pl.read_csv(
        DATASET_PATH, schema_overrides={"ocean_proximity": ocean_proximity_enum}
    )
    data_complete: pl.DataFrame = data.filter(
        ~(
            pl.any_horizontal(pl.all().is_null())
            | pl.any_horizontal(cs.float().is_nan())
        )
    )

    logger.info("Formato dos dados: %s", data.shape)
    logger.info("Formato dos dados completos: %s", data_complete.shape)
    logger.info(f"%s{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}", data_complete.head())

    return data_complete


def dendrogram_plot(linkage_matrix: FloatArray, linkage_method: str) -> None:
    logger.info("EXECUTANDO dendrogram_plot para ligação %s", linkage_method)

    with subplots(
        nrows=1,
        ncols=1,
        figsize=(18, 10),
        savefig_path=CLUSTERING_OUTPUT_DIRECTORY_PATH
        / f"median_income_{linkage_method}_dendrograma.png",
    ) as dendrogram_ax:
        dendrogram(
            linkage_matrix,
            no_labels=True,
            ax=dendrogram_ax,
        )
        dendrogram_ax.set_title(f"Dendrograma de median_income ({linkage_method})")
        dendrogram_ax.set_xlabel("Observações")
        dendrogram_ax.set_ylabel("Distância")

    logger.info(
        f"dendrogram_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def boxplot_cluster_plot(
    data: pl.DataFrame,
    clusters: IntArray,
    linkage_method: str,
    cluster_count: int,
) -> None:
    logger.info(
        "EXECUTANDO boxplot_cluster_plot para ligação %s com %s clusters",
        linkage_method,
        cluster_count,
    )
    data_clustered: pl.DataFrame = data.with_columns(
        pl.Series(MEDIAN_INCOME_CLUSTER_COLUMN_NAME, clusters)
    )

    with subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 12),
        savefig_path=CLUSTERING_OUTPUT_DIRECTORY_PATH
        / f"median_income_{linkage_method}_{cluster_count}_clusters_boxplot.png",
    ) as boxplot_ax:
        sns.boxplot(
            data=data_clustered,
            x=MEDIAN_INCOME_CLUSTER_COLUMN_NAME,
            y=MEDIAN_INCOME_COLUMN_NAME,
            ax=boxplot_ax,
        )
        boxplot_ax.set_title(
            f"Boxplot de median_income por cluster ({linkage_method}, {cluster_count} clusters)"
        )

    logger.info(
        f"boxplot_cluster_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def best_clusters_find(
    data: pl.DataFrame,
    observations: FloatArray,
    linkage_method: str,
    linkage_matrix: FloatArray,
) -> tuple[IntArray, int, float]:
    best_clusters: IntArray = np.array([], dtype=np.int32)
    best_cluster_count: int = 0
    best_silhouette_score: float = -np.inf

    for cluster_count in CLUSTER_COUNTS:
        clusters: IntArray = np.asarray(
            fcluster(linkage_matrix, t=cluster_count, criterion="maxclust"),
            dtype=np.int32,
        )
        # False positive: sklearn accepts 2D NumPy arrays for X, but some
        # Pyrefly/sklearn stub combinations reject parameterized NDArray.
        clusters_silhouette_score: float = float(
            silhouette_score(cast(Any, observations), clusters)
        )
        logger.info(
            "Ligação %s com %s clusters: silhouette = %.6f",
            linkage_method,
            cluster_count,
            clusters_silhouette_score,
        )

        if clusters_silhouette_score > best_silhouette_score:
            best_clusters = clusters
            best_cluster_count = cluster_count
            best_silhouette_score = clusters_silhouette_score

    boxplot_cluster_plot(data, best_clusters, linkage_method, best_cluster_count)

    return best_clusters, best_cluster_count, best_silhouette_score


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(module)s.%(funcName)s:%(message)s"
    )
    sns.set_theme()

    reset_directory(CLUSTERING_OUTPUT_DIRECTORY_PATH, "saídas de clusterização")
    data: pl.DataFrame = data_complete_load()
    median_income_values: FloatArray = np.asarray(
        data[MEDIAN_INCOME_COLUMN_NAME].to_numpy(), dtype=np.float64
    )
    median_income_observations: FloatArray = median_income_values.reshape(-1, 1)

    best_linkage_method: str = ""
    best_clusters: IntArray = np.array([], dtype=np.int32)
    best_cluster_count: int = 0
    best_silhouette_score: float = -np.inf

    for linkage_method in LINKAGE_METHODS:
        logger.info("EXECUTANDO linkage para ligação %s", linkage_method)
        linkage_matrix: FloatArray = linkage(
            median_income_observations, method=linkage_method
        )
        logger.info(
            f"linkage executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
        )

        dendrogram_plot(linkage_matrix, linkage_method)
        linkage_clusters: IntArray
        cluster_count: int
        silhouette_score: float
        linkage_clusters, cluster_count, silhouette_score = best_clusters_find(
            data, median_income_observations, linkage_method, linkage_matrix
        )

        if silhouette_score > best_silhouette_score:
            best_linkage_method = linkage_method
            best_clusters = linkage_clusters
            best_cluster_count = cluster_count
            best_silhouette_score = silhouette_score

    data.with_columns(
        pl.Series(MEDIAN_INCOME_CLUSTER_COLUMN_NAME, best_clusters)
    ).write_csv(HOUSING_STRATIFIED_PATH)

    logger.info(
        "Melhor clusterização: ligação %s, %s clusters, silhouette = %.6f",
        best_linkage_method,
        best_cluster_count,
        best_silhouette_score,
    )
    logger.info("Arquivo salvo em %s", HOUSING_STRATIFIED_PATH)


if __name__ == "__main__":
    main()
