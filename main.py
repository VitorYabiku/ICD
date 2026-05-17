import polars.selectors as pls
from pathlib import Path

import altair as alt
import polars as pl


def main():
    DATASET_PATH = Path("dataset/housing.csv")
    data = pl.scan_csv(DATASET_PATH).collect()

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
        "Amplitude": data.select(pls.numeric().max() - pls.numeric().min()),
    }

    STATS_DESCRIPTIVE_COLUMN_NAME = "Estatística"
    for stat_name, stat_lazyframe in stats_descriptive.items():
        stat_dataframe = stat_lazyframe.with_columns(
            pl.lit(stat_name).alias(STATS_DESCRIPTIVE_COLUMN_NAME)
        ).collect()

        print(stat_dataframe)

    value_histogram = data.collect().plot.bar(
        x=alt.X(
            "total_rooms:N",
            bin=alt.Bin(step=500),
            title="Quantidade total de quartos no bloco",
        ),
        y=alt.Y("count()", title="Número de blocos"),
    )

    value_histogram.save("plots/histogram.png")

    value_kernel_density_estimator = (
        alt.Chart(data.collect())
        .transform_density(
            "total_rooms",
            as_=["total_rooms", "density"],
        )
        .mark_line(color="darkred")
        .encode(
            x="total_rooms:N",
            y=alt.Y("density:Q", title="Densidade"),
        )
    )

    value_kernel_density_estimator.save("plots/kernel_density_estimator.png")

    # (value_histogram + value_kernel_density_estimator).save("plots/test.png")


if __name__ == "__main__":
    main()
