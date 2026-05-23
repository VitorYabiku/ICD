import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
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
MEDIAN_INCOME_CLUSTER_COLUMN_NAME: str = "cluster"
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


def data_complete_load() -> pl.LazyFrame:
    ocean_proximity_enum: pl.Enum = pl.Enum(
        OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING
    )
    data: pl.LazyFrame = pl.scan_csv(
        DATASET_PATH, schema_overrides={"ocean_proximity": ocean_proximity_enum}
    )
    data_filtered: pl.LazyFrame = data.drop_nulls().drop_nans()
    logger.info(
        "Quantidade de observações dos dados originais: %s", data.collect_schema().len()
    )
    logger.info(
        "Quantidade de observações dos dados filtrados: %s",
        data_filtered.collect_schema().len(),
    )
    logger.info(f"%s{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}", data_filtered.head())

    return data_filtered


SilhouetteScores = list[tuple[int, float]]


def clustering_summary_plot(
    data: pl.LazyFrame,
    clusters: IntArray,
    linkage_method: str,
    cluster_count: int,
    linkage_matrix: FloatArray,
    silhouette_scores: SilhouetteScores,
) -> None:
    logger.info(
        "EXECUTANDO clustering_summary_plot para ligação %s com %s clusters",
        linkage_method,
        cluster_count,
    )
    cluster_order: list[int] = sorted(int(cluster) for cluster in np.unique(clusters))
    data_clustered: pl.DataFrame = data.with_columns(
        pl.Series(MEDIAN_INCOME_CLUSTER_COLUMN_NAME, clusters)
    ).collect()
    cluster_silhouette_per_count: list[int] = [
        cluster_count for cluster_count, _ in silhouette_scores
    ]
    silhouette_score_values: list[float] = [
        silhouette_score for _, silhouette_score in silhouette_scores
    ]

    with subplots(
        nrows=3,
        ncols=1,
        figsize=(18, 24),
        layout="constrained",
        savefig_path=CLUSTERING_OUTPUT_DIRECTORY_PATH
        / f"median_income_{linkage_method}_clusterizacao.png",
    ) as (dendrogram_ax, silhouette_ax, boxplot_ax):
        dendrogram(
            linkage_matrix,
            no_labels=True,
            ax=dendrogram_ax,
        )
        dendrogram_ax.set_title(
            f"Dendrograma de median_income (método de ligação = {linkage_method})"
        )
        dendrogram_ax.set_xlabel("Observações")
        dendrogram_ax.set_ylabel("Distância")

        sns.lineplot(
            x=cluster_silhouette_per_count,
            y=silhouette_score_values,
            marker="o",
            ax=silhouette_ax,
        )
        silhouette_ax.axvline(
            cluster_count,
            color="red",
            linestyle="--",
            linewidth=1,
        )
        silhouette_ax.set_title(
            f"Silhueta por quantidade de clusters (método de ligação = {linkage_method})"
        )
        silhouette_ax.set_xlabel("Quantidade de clusters")
        silhouette_ax.set_ylabel("Silhueta")
        silhouette_ax.set_xticks(cluster_silhouette_per_count)

        sns.boxplot(
            data=data_clustered,
            x=MEDIAN_INCOME_CLUSTER_COLUMN_NAME,
            y=MEDIAN_INCOME_COLUMN_NAME,
            order=cluster_order,
            ax=boxplot_ax,
        )
        boxplot_ax.set_title(
            f"Boxplots de median_income por cluster (método de ligação = {linkage_method}, nº de clusters = {cluster_count})"
        )
        boxplot_ax.set_xlabel(MEDIAN_INCOME_CLUSTER_COLUMN_NAME)
        boxplot_ax.set_ylabel(MEDIAN_INCOME_COLUMN_NAME)

    logger.info(
        f"clustering_summary_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def best_clustering_summary_plot(
    data: pl.LazyFrame,
    linkage_method: str,
    best_silhouette_score: float,
) -> None:
    logger.info(
        "EXECUTANDO best_clustering_summary_plot para ligação %s",
        linkage_method,
    )

    clusters = (
        data.select(pl.col(MEDIAN_INCOME_CLUSTER_COLUMN_NAME))
        .collect()
        .get_column(MEDIAN_INCOME_CLUSTER_COLUMN_NAME)
        .to_numpy()
    )
    clusters_unique = np.unique(clusters)
    cluster_count = len(clusters_unique)
    cluster_order: list[int] = sorted(int(cluster) for cluster in clusters_unique)
    # median_income_values: FloatArray = np.asarray(clusters, dtype=np.float64)
    # median_income_min: float = float(median_income_values.min())
    # median_income_max: float = float(median_income_values.max())

    with subplots(
        nrows=3,
        ncols=1,
        figsize=(18, 24),
        layout="constrained",
        savefig_path=CLUSTERING_OUTPUT_DIRECTORY_PATH
        / "median_income_best_clusterizacao_resumo.png",
    ) as (distribution_ax, boxplot_ax, cluster_size_ax):
        sns.histplot(
            data=data.collect(),
            x=MEDIAN_INCOME_COLUMN_NAME,
            hue=MEDIAN_INCOME_CLUSTER_COLUMN_NAME,
            hue_order=cluster_order,
            multiple="stack",
            bins=30,
            ax=distribution_ax,
        )
        for container in distribution_ax.containers:
            distribution_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
        distribution_ax.set_title(
            "Histograma de median_income por cluster "
            f"({linkage_method}, {cluster_count} clusters, "
            f"silhueta = {best_silhouette_score:.3f})"
        )
        # distribution_ax.set_xlabel("Observações")
        # distribution_ax.set_ylabel(MEDIAN_INCOME_COLUMN_NAME)
        # distribution_ax.set_ylim(median_income_min, median_income_max)

        sns.boxplot(
            data=data.collect(),
            x=MEDIAN_INCOME_CLUSTER_COLUMN_NAME,
            y=MEDIAN_INCOME_COLUMN_NAME,
            order=cluster_order,
            ax=boxplot_ax,
        )
        boxplot_ax.set_title(
            f"Boxplots de median_income por cluster (método de ligação = {linkage_method}, nº de clusters = {cluster_count})"
        )
        # boxplot_ax.set_xlabel(MEDIAN_INCOME_CLUSTER_COLUMN_NAME)
        # boxplot_ax.set_ylabel(MEDIAN_INCOME_COLUMN_NAME)
        # boxplot_ax.set_ylim(median_income_min, median_income_max)

        sns.countplot(
            data=data.collect(),
            x=MEDIAN_INCOME_CLUSTER_COLUMN_NAME,
            order=cluster_order,
            stat="percent",
            ax=cluster_size_ax,
        )
        for container in cluster_size_ax.containers:
            cluster_size_ax.bar_label(container, fmt="%.0f", padding=3, fontsize=16)
        cluster_size_ax.set_title("Proporção de observações por cluster")
        # cluster_size_ax.set_xlabel(MEDIAN_INCOME_CLUSTER_COLUMN_NAME)
        # cluster_size_ax.set_ylabel("% de observações")

    logger.info(
        f"best_clustering_summary_plot executado com SUCESSO{LOG_SPACING_VERTICAL_LINE_COUNT * '\n'}"
    )


def best_clusters_find(
    observations: FloatArray,
    linkage_method: str,
    linkage_matrix: FloatArray,
) -> tuple[IntArray, int, float, SilhouetteScores]:
    best_clusters: IntArray = np.array([], dtype=np.int32)
    best_cluster_count: int = 0
    best_silhouette_score: float = -np.inf
    silhouette_scores: SilhouetteScores = []

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
            "Ligação %s com %s clusters: silhueta = %.3f",
            linkage_method,
            cluster_count,
            clusters_silhouette_score,
        )
        silhouette_scores.append((cluster_count, clusters_silhouette_score))

        if clusters_silhouette_score > best_silhouette_score:
            best_clusters = clusters
            best_cluster_count = cluster_count
            best_silhouette_score = clusters_silhouette_score

    return best_clusters, best_cluster_count, best_silhouette_score, silhouette_scores


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(module)s.%(funcName)s:%(message)s"
    )
    sns.set_theme()

    reset_directory(CLUSTERING_OUTPUT_DIRECTORY_PATH, "saídas de clusterização")
    data: pl.LazyFrame = data_complete_load()
    median_income_values: FloatArray = np.asarray(
        data.select(pl.col(MEDIAN_INCOME_COLUMN_NAME))
        .collect()
        .get_column(MEDIAN_INCOME_COLUMN_NAME)
        .to_numpy(),
        dtype=np.float64,
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

        linkage_clusters: IntArray
        cluster_count: int
        silhouette_score: float
        silhouette_scores: SilhouetteScores
        (
            linkage_clusters,
            cluster_count,
            silhouette_score,
            silhouette_scores,
        ) = best_clusters_find(
            median_income_observations, linkage_method, linkage_matrix
        )

        clustering_summary_plot(
            data,
            linkage_clusters,
            linkage_method,
            cluster_count,
            linkage_matrix,
            silhouette_scores,
        )

        if silhouette_score > best_silhouette_score:
            best_linkage_method = linkage_method
            best_clusters = linkage_clusters
            best_cluster_count = cluster_count
            best_silhouette_score = silhouette_score

    data = data.with_columns(
        pl.Series(MEDIAN_INCOME_CLUSTER_COLUMN_NAME, best_clusters)
    )

    best_clustering_summary_plot(
        data,
        best_linkage_method,
        best_silhouette_score,
    )

    data.collect().write_csv(HOUSING_STRATIFIED_PATH)

    logger.info(
        "Melhor clusterização: ligação %s, %s clusters, silhueta = %.3f",
        best_linkage_method,
        best_cluster_count,
        best_silhouette_score,
    )
    logger.info("Arquivo salvo em %s", HOUSING_STRATIFIED_PATH)


if __name__ == "__main__":
    main()
