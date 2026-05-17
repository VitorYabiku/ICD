import polars.selectors as cs
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
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

    COLUMNS_EXCLUDED = ["longitude", "latitude"]
    histogram_data = data.select(cs.numeric().exclude(COLUMNS_EXCLUDED)).collect()
    for column_name in histogram_data.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        sns.histplot(histogram_data, x=column_name, kde=True, ax=ax1)

        sns.boxplot(histogram_data, x=column_name, ax=ax2)

        # sns.ecdfplot(histogram_data, x=column_name, ax=ax3)

        fig.savefig(PLOT_DIRECTORY_PATH / f"{column_name}_histogram_&_boxplot.png")
        plt.close()


if __name__ == "__main__":
    main()
