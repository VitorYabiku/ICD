import json
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl
import polars.selectors as cs
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

EP3_DIRECTORY_PATH: Final[Path] = Path(__file__).resolve().parent

CROSS_VALIDATION_FOLD_COUNT: Final[int] = 10
RANDOM_STATE: Final[int] = 42
CROSS_VALIDATION_SCORING: Final[str] = "r2"
GRID_SEARCH_N_JOBS: Final[int] = -1
type Hyperparameters = dict[str, object]
type EvaluationScores = dict[str, list[float]]
type HyperparameterSelection = dict[str, object]
type TrainedModel = Pipeline | DecisionTreeRegressor | RandomForestRegressor
type FeatureImportance = dict[str, object]

TARGET_VARIABLE_COLUMN_NAME: Final = "median_income"
CLUSTER_COLUMN_NAME: Final = "cluster"
NUMERIC_FEATURE_COLUMN_NAMES: Final = (
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
)
INNER_CROSS_VALIDATION: Final = StratifiedKFold(
    n_splits=CROSS_VALIDATION_FOLD_COUNT,
    shuffle=True,
    random_state=RANDOM_STATE,
)
OUTER_CROSS_VALIDATION: Final = StratifiedKFold(
    n_splits=CROSS_VALIDATION_FOLD_COUNT,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def outlier_expr(column_name: str) -> pl.Expr:
    first_quartile = pl.col(column_name).quantile(0.25)
    third_quartile = pl.col(column_name).quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    OUTLIER_INTERQUARTILE_RANGE_FACTOR: Final = 1.5
    return (
        pl.col(column_name)
        < first_quartile - OUTLIER_INTERQUARTILE_RANGE_FACTOR * interquartile_range
    ) | (
        pl.col(column_name)
        > third_quartile + OUTLIER_INTERQUARTILE_RANGE_FACTOR * interquartile_range
    )


def features_target_and_clusters_get(
    dataset: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = dataset.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
    clusters = dataset.get_column(CLUSTER_COLUMN_NAME).to_numpy()
    features = dataset.drop(TARGET_VARIABLE_COLUMN_NAME, CLUSTER_COLUMN_NAME).to_numpy()
    return features, target, clusters


def error_score_format(score: float, target_mean: float) -> str:
    return f"{score:.4f} ({score / target_mean:.2%} da média da coluna alvo)"


def evaluation_result_get(
    scores: EvaluationScores, target_mean: float
) -> dict[str, object]:
    folds = [
        {
            "fold": fold,
            "r2": r2,
            "rmse": {"valor": rmse, "proporcao_media_alvo": rmse / target_mean},
            "mae": {"valor": mae, "proporcao_media_alvo": mae / target_mean},
        }
        for fold, (r2, rmse, mae) in enumerate(
            zip(scores["r2"], scores["rmse"], scores["mae"], strict=True), start=1
        )
    ]
    mean_rmse = float(np.mean(scores["rmse"]))
    mean_mae = float(np.mean(scores["mae"]))
    return {
        "folds": folds,
        "medias": {
            "r2": float(np.mean(scores["r2"])),
            "tempo_de_treinamento_segundos": float(
                np.mean(scores["tempo_de_treinamento_segundos"])
            ),
            "rmse": {
                "valor": mean_rmse,
                "proporcao_media_alvo": mean_rmse / target_mean,
            },
            "mae": {
                "valor": mean_mae,
                "proporcao_media_alvo": mean_mae / target_mean,
            },
        },
    }


def model_evaluation_print(
    model_name: str,
    scores: EvaluationScores,
    target_mean: float,
    *,
    display_folds: bool = True,
) -> None:
    print(f"\n--- Avaliação: {model_name} ---")
    if display_folds:
        print("\nAs métricas externas abaixo foram calculadas em cada fold externa.")
        for fold, (r2, rmse, mae) in enumerate(
            zip(scores["r2"], scores["rmse"], scores["mae"], strict=True), start=1
        ):
            print(
                f"Fold {fold}: R² = {r2:.4f}; "
                f"RMSE = {error_score_format(rmse, target_mean)}; "
                f"MAE = {error_score_format(mae, target_mean)}"
            )
    print(f"R² médio: {np.mean(scores['r2']):.4f}")
    print(
        "Tempo médio de treinamento: "
        f"{np.mean(scores['tempo_de_treinamento_segundos']):.4f} s"
    )
    for metric, label in (("rmse", "RMSE"), ("mae", "MAE")):
        mean_score = float(np.mean(scores[metric]))
        print(f"{label} médio: {error_score_format(mean_score, target_mean)}")


def feature_importance_get(
    models: list[TrainedModel], feature_names: list[str]
) -> FeatureImportance:
    fold_importance_values: list[np.ndarray] = []
    folds: list[dict[str, object]] = []
    method = "coeficientes_absolutos"
    for fold, model in enumerate(models, start=1):
        # Only linear regression uses a Pipeline because it includes feature scaling.
        if isinstance(model, Pipeline):
            linear_model = cast(LinearRegression, model.named_steps["model"])
            importance_values = np.abs(linear_model.coef_)
        else:
            method = "importancia_por_impureza"
            importance_values = model.feature_importances_
        fold_importance_values.append(importance_values)
        folds.append(
            {
                "fold": fold,
                "features": [
                    {"feature": feature_name, "importancia": float(importance)}
                    for feature_name, importance in sorted(
                        zip(feature_names, importance_values, strict=True),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ],
            }
        )

    overall_importance_values = np.mean(fold_importance_values, axis=0)
    return {
        "metodo": method,
        "folds": folds,
        "media_geral": [
            {"feature": feature_name, "importancia": float(importance)}
            for feature_name, importance in sorted(
                zip(feature_names, overall_importance_values, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


def feature_importance_print(
    model_name: str, feature_importance: FeatureImportance
) -> None:
    print(f"\n=== Importância das features: {model_name} ===")
    for fold in cast(list[dict[str, object]], feature_importance["folds"]):
        print(f"Fold {fold['fold']}:")
        for feature in cast(list[dict[str, object]], fold["features"]):
            print(f"  {feature['feature']}: {feature['importancia']:.6f}")
    print("Média geral:")
    for feature in cast(list[dict[str, object]], feature_importance["media_geral"]):
        print(f"  {feature['feature']}: {feature['importancia']:.6f}")


def model_bundle_save(
    dataset_key: str,
    dataset_name: str,
    model_key: str,
    model_name: str,
    feature_names: list[str],
    models: list[TrainedModel],
    evaluation: dict[str, object],
    feature_importance: FeatureImportance,
) -> Path:
    MODEL_OUTPUT_DIRECTORY_PATH: Final = EP3_DIRECTORY_PATH / "output" / "modelos"
    MODEL_OUTPUT_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_OUTPUT_DIRECTORY_PATH / f"{dataset_key}_{model_key}.joblib"
    temporary_output_path = output_path.with_suffix(".joblib.tmp")
    dump(
        {
            "valor_chave_do_conjunto_de_dados": dataset_key,
            "conjunto_de_dados": dataset_name,
            "valor_chave_do_modelo": model_key,
            "modelo": model_name,
            "nomes_das_features": feature_names,
            "modelos": [
                {"fold": fold, "estimador": model}
                for fold, model in enumerate(models, start=1)
            ],
            "avaliacao": evaluation,
            "importancia_das_features": feature_importance,
        },
        temporary_output_path,
    )
    temporary_output_path.replace(output_path)
    return output_path


def linear_regression_train(
    dataset_lazy: pl.LazyFrame,
    target_mean: float,
) -> tuple[EvaluationScores, list[TrainedModel]]:
    print("\n=== Regressão linear ===")
    print("--- Treinamento: Regressão linear ---")
    dataset = dataset_lazy.collect()
    features, target, clusters = features_target_and_clusters_get(dataset)
    LINEAR_REGRESSION_NUMERIC_FEATURE_INDICES: Final = tuple(
        range(len(NUMERIC_FEATURE_COLUMN_NAMES))
    )
    model: Final = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            StandardScaler(),
                            LINEAR_REGRESSION_NUMERIC_FEATURE_INDICES,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    cross_validation_results = cross_validate(
        model,
        features,
        target,
        cv=OUTER_CROSS_VALIDATION.split(features, clusters),
        scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
        return_estimator=True,
    )
    scores: EvaluationScores = {
        "r2": [float(score) for score in cross_validation_results["test_r2"]],
        "tempo_de_treinamento_segundos": [
            float(time) for time in cross_validation_results["fit_time"]
        ],
        "rmse": [
            -float(score)
            for score in cross_validation_results["test_neg_root_mean_squared_error"]
        ],
        "mae": [
            -float(score)
            for score in cross_validation_results["test_neg_mean_absolute_error"]
        ],
    }

    model_evaluation_print("Regressão linear", scores, target_mean)

    return scores, [
        cast(Pipeline, estimator) for estimator in cross_validation_results["estimator"]
    ]


def decision_tree_train(
    dataset_lazy: pl.LazyFrame,
    target_mean: float,
) -> tuple[
    EvaluationScores,
    Hyperparameters,
    HyperparameterSelection,
    list[TrainedModel],
]:
    dataset = dataset_lazy.collect()
    features, target, clusters = features_target_and_clusters_get(dataset)
    print("\n=== Árvore de decisão ===")
    print("--- Treinamento e seleção de hiperparâmetros: Árvore de decisão ---")
    scores: EvaluationScores = {
        "r2": [],
        "tempo_de_treinamento_segundos": [],
        "rmse": [],
        "mae": [],
    }
    candidate_mean_scores_by_fold: list[np.ndarray] = []
    selection_by_fold: list[dict[str, object]] = []
    models: list[TrainedModel] = []
    candidate_params: np.ndarray = np.array([], dtype=object)
    DECISION_TREE_MAX_DEPTH_VALUES: Final = (None, 3, 5, 10, 20)
    DECISION_TREE_MIN_SAMPLES_SPLIT_VALUES: Final = (2, 5, 10)
    DECISION_TREE_MIN_SAMPLES_LEAF_VALUES: Final = (1, 2, 4)
    DECISION_TREE_PARAM_GRID: Final = {
        "max_depth": DECISION_TREE_MAX_DEPTH_VALUES,
        "min_samples_split": DECISION_TREE_MIN_SAMPLES_SPLIT_VALUES,
        "min_samples_leaf": DECISION_TREE_MIN_SAMPLES_LEAF_VALUES,
    }
    for fold, (train_indices, test_indices) in enumerate(
        OUTER_CROSS_VALIDATION.split(features, clusters), start=1
    ):
        print(f"Processando fold externa {fold}/{CROSS_VALIDATION_FOLD_COUNT}...")
        search = GridSearchCV(
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            DECISION_TREE_PARAM_GRID,
            cv=list(
                INNER_CROSS_VALIDATION.split(
                    features[train_indices], clusters[train_indices]
                )
            ),
            scoring=CROSS_VALIDATION_SCORING,
            n_jobs=GRID_SEARCH_N_JOBS,
        )
        search.fit(features[train_indices], target[train_indices])
        scores["tempo_de_treinamento_segundos"].append(search.refit_time_)
        candidate_mean_scores_by_fold.append(search.cv_results_["mean_test_score"])
        candidate_params = np.asarray(search.cv_results_["params"], dtype=object)
        best_params = search.best_params_
        models.append(search.best_estimator_)
        selection_by_fold.append(
            {
                "fold": fold,
                "melhores_hiperparametros": dict(best_params),
            }
        )
        predictions = search.predict(features[test_indices])
        r2 = float(search.score(features[test_indices], target[test_indices]))
        rmse = root_mean_squared_error(target[test_indices], predictions)
        mae = mean_absolute_error(target[test_indices], predictions)
        scores["r2"].append(r2)
        scores["rmse"].append(rmse)
        scores["mae"].append(mae)
        print(
            f"Melhores hiperparâmetros da fold externa {fold}: "
            f"max_depth={best_params['max_depth']}, "
            f"min_samples_split={best_params['min_samples_split']}, "
            f"min_samples_leaf={best_params['min_samples_leaf']}; "
            f"R² externo={r2:.4f}; "
            f"RMSE externo={error_score_format(rmse, target_mean)}; "
            f"MAE externo={error_score_format(mae, target_mean)}"
        )

    overall_candidate_mean_scores = np.mean(candidate_mean_scores_by_fold, axis=0)
    overall_best_candidate_index = int(np.argmax(overall_candidate_mean_scores))
    overall_best_params = candidate_params[overall_best_candidate_index]
    print(
        "\nOs R² externos abaixo pertencem à melhor combinação de cada fold "
        "externa, não a uma única combinação geral."
    )
    model_evaluation_print(
        "Árvore de decisão", scores, target_mean, display_folds=False
    )
    print(
        "Melhor combinação geral por R² médio interno: "
        f"max_depth={overall_best_params['max_depth']}, "
        f"min_samples_split={overall_best_params['min_samples_split']}, "
        f"min_samples_leaf={overall_best_params['min_samples_leaf']}; "
        "R² médio interno entre folds externas="
        f"{overall_candidate_mean_scores[overall_best_candidate_index]:.4f}"
    )

    return (
        scores,
        dict(overall_best_params),
        {
            "folds": selection_by_fold,
            "melhores_hiperparametros_gerais": dict(overall_best_params),
            "r2_medio_interno_geral": float(
                overall_candidate_mean_scores[overall_best_candidate_index]
            ),
        },
        models,
    )


def random_forest_train(
    dataset_lazy: pl.LazyFrame,
    target_mean: float,
) -> tuple[
    EvaluationScores,
    Hyperparameters,
    HyperparameterSelection,
    list[TrainedModel],
]:
    dataset = dataset_lazy.collect()
    features, target, clusters = features_target_and_clusters_get(dataset)
    print("\n=== Floresta aleatória ===")
    print("--- Treinamento e seleção de hiperparâmetros: Floresta aleatória ---")
    scores: EvaluationScores = {
        "r2": [],
        "tempo_de_treinamento_segundos": [],
        "rmse": [],
        "mae": [],
    }
    candidate_mean_scores_by_fold: list[np.ndarray] = []
    selection_by_fold: list[dict[str, object]] = []
    models: list[TrainedModel] = []
    candidate_params: np.ndarray = np.array([], dtype=object)
    RANDOM_FOREST_N_JOBS: Final = 1
    RANDOM_FOREST_N_ESTIMATORS_VALUES: Final = (50, 100, 200, 400)
    RANDOM_FOREST_MAX_DEPTH_VALUES: Final = (None, 5, 10, 20)
    RANDOM_FOREST_MAX_FEATURES_VALUES: Final = (1.0, "sqrt", "log2")
    RANDOM_FOREST_PARAM_GRID: Final = {
        "n_estimators": RANDOM_FOREST_N_ESTIMATORS_VALUES,
        "max_depth": RANDOM_FOREST_MAX_DEPTH_VALUES,
        "max_features": RANDOM_FOREST_MAX_FEATURES_VALUES,
    }
    for fold, (train_indices, test_indices) in enumerate(
        OUTER_CROSS_VALIDATION.split(features, clusters), start=1
    ):
        print(f"Processando fold externa {fold}/{CROSS_VALIDATION_FOLD_COUNT}...")
        search = GridSearchCV(
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=RANDOM_FOREST_N_JOBS,
            ),
            RANDOM_FOREST_PARAM_GRID,
            cv=list(
                INNER_CROSS_VALIDATION.split(
                    features[train_indices], clusters[train_indices]
                )
            ),
            scoring=CROSS_VALIDATION_SCORING,
            n_jobs=GRID_SEARCH_N_JOBS,
        )
        search.fit(features[train_indices], target[train_indices])
        scores["tempo_de_treinamento_segundos"].append(search.refit_time_)
        candidate_mean_scores_by_fold.append(search.cv_results_["mean_test_score"])
        candidate_params = np.asarray(search.cv_results_["params"], dtype=object)
        best_params = search.best_params_
        models.append(search.best_estimator_)
        selection_by_fold.append(
            {
                "fold": fold,
                "melhores_hiperparametros": dict(best_params),
            }
        )
        predictions = search.predict(features[test_indices])
        r2 = float(search.score(features[test_indices], target[test_indices]))
        rmse = root_mean_squared_error(target[test_indices], predictions)
        mae = mean_absolute_error(target[test_indices], predictions)
        scores["r2"].append(r2)
        scores["rmse"].append(rmse)
        scores["mae"].append(mae)
        print(
            f"Melhores hiperparâmetros da fold externa {fold}: "
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, "
            f"max_features={best_params['max_features']}; "
            f"R² externo={r2:.4f}; "
            f"RMSE externo={error_score_format(rmse, target_mean)}; "
            f"MAE externo={error_score_format(mae, target_mean)}"
        )

    overall_candidate_mean_scores = np.mean(candidate_mean_scores_by_fold, axis=0)
    overall_best_candidate_index = int(np.argmax(overall_candidate_mean_scores))
    overall_best_params = candidate_params[overall_best_candidate_index]
    model_evaluation_print(
        "Floresta aleatória", scores, target_mean, display_folds=False
    )
    print(
        "Melhor combinação geral por R² médio interno: "
        f"n_estimators={overall_best_params['n_estimators']}, "
        f"max_depth={overall_best_params['max_depth']}, "
        f"max_features={overall_best_params['max_features']}; "
        "R² médio interno entre folds externas="
        f"{overall_candidate_mean_scores[overall_best_candidate_index]:.4f}"
    )

    return (
        scores,
        dict(overall_best_params),
        {
            "folds": selection_by_fold,
            "melhores_hiperparametros_gerais": dict(overall_best_params),
            "r2_medio_interno_geral": float(
                overall_candidate_mean_scores[overall_best_candidate_index]
            ),
        },
        models,
    )


def models_comparison_print(
    model_scores: list[tuple[str, EvaluationScores]],
    target_mean: float,
) -> None:
    print("\n=== Comparação final ===")
    for model_name, scores in sorted(
        model_scores, key=lambda item: np.mean(item[1]["r2"]), reverse=True
    ):
        mean_rmse = float(np.mean(scores["rmse"]))
        mean_mae = float(np.mean(scores["mae"]))
        print(
            f"{model_name}: "
            f"R² médio={np.mean(scores['r2']):.4f}; "
            f"RMSE médio={error_score_format(mean_rmse, target_mean)}; "
            f"MAE médio={error_score_format(mean_mae, target_mean)}"
        )


def hyperparameter_comparison_print(
    best_hyperparameters_by_dataset: dict[str, dict[str, Hyperparameters]],
) -> None:
    print("\n=== Sensibilidade dos hiperparâmetros a outliers ===")
    print("Legenda: com outliers → sem outliers; = igual; ≠ diferente")
    with_outliers = best_hyperparameters_by_dataset["Dados com outliers"]
    without_outliers = best_hyperparameters_by_dataset["Dados sem outliers"]
    for model_name, model_hyperparameters_with_outliers in with_outliers.items():
        print(f"{model_name}:")
        model_hyperparameters_without_outliers = without_outliers[model_name]
        for (
            hyperparameter_name,
            value_with_outliers,
        ) in model_hyperparameters_with_outliers.items():
            value_without_outliers = model_hyperparameters_without_outliers[
                hyperparameter_name
            ]
            symbol = "=" if value_with_outliers == value_without_outliers else "≠"
            print(
                f"  {hyperparameter_name}: {value_with_outliers} → "
                f"{value_without_outliers} {symbol}"
            )


def main() -> None:
    PROJECT_DIRECTORY_PATH: Final = EP3_DIRECTORY_PATH.parent
    DATASET_DIRECTORY_PATH: Final = PROJECT_DIRECTORY_PATH / "dataset"
    DATASET_PATH: Final = DATASET_DIRECTORY_PATH / "housing_stratified.csv"
    OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING: Final = (
        "ISLAND",
        "NEAR OCEAN",
        "NEAR BAY",
        "<1H OCEAN",
        "INLAND",
    )
    USED_VARIABLE_COLUMN_NAMES: Final = (
        TARGET_VARIABLE_COLUMN_NAME,
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "ocean_proximity",
        CLUSTER_COLUMN_NAME,
    )
    dataset_lazy: pl.LazyFrame = pl.scan_csv(
        DATASET_PATH,
        schema_overrides={
            "ocean_proximity": pl.Enum(OCEAN_PROXIMITY_CATEGORIES_ORDERED_ASCENDING)
        },
    ).select(*USED_VARIABLE_COLUMN_NAMES)

    ROW_WITH_NULL_OR_NAN_EXPR: Final[pl.Expr] = pl.any_horizontal(
        pl.all().is_null()
    ) | pl.any_horizontal(cs.float().is_nan())
    total_row_count = dataset_lazy.select(pl.len()).collect().item()
    invalid_row_count = (
        dataset_lazy.filter(ROW_WITH_NULL_OR_NAN_EXPR).select(pl.len()).collect().item()
    )
    print("=== Preparação dos dados ===")
    print(f"Linhas totais: {total_row_count}")
    print(f"Linhas removidas por null ou NaN: {invalid_row_count}")
    print(f"Linhas utilizadas: {total_row_count - invalid_row_count}")

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
            CLUSTER_COLUMN_NAME,
        )
        .collect()
    )
    ROW_WITH_OUTLIER_EXPR: Final[pl.Expr] = pl.any_horizontal(
        *[
            outlier_expr(column_name)
            for column_name in (
                TARGET_VARIABLE_COLUMN_NAME,
                *NUMERIC_FEATURE_COLUMN_NAMES,
            )
        ]
    )
    dataset_without_outliers = dataset.filter(~ROW_WITH_OUTLIER_EXPR)
    outlier_row_count = dataset.height - dataset_without_outliers.height
    print(
        "Linhas removidas por um ou mais outliers: "
        f"{outlier_row_count} ({outlier_row_count / dataset.height:.2%})"
    )
    print(f"Linhas utilizadas sem outliers: {dataset_without_outliers.height}")

    results: dict[str, object] = {
        "preparacao_dos_dados": {
            "linhas_totais": total_row_count,
            "linhas_removidas_por_nulo_ou_nan": invalid_row_count,
            "linhas_utilizadas": total_row_count - invalid_row_count,
            "linhas_removidas_por_outliers": {
                "quantidade": outlier_row_count,
                "proporcao": outlier_row_count / dataset.height,
            },
            "linhas_utilizadas_sem_outliers": dataset_without_outliers.height,
        }
    }
    datasets_results: dict[str, object] = {}
    mean_scores_by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    target_means_by_dataset: dict[str, float] = {}
    best_hyperparameters_by_dataset: dict[str, dict[str, Hyperparameters]] = {}
    TARGET_VARIABLE_MEASUREMENT_UNIT: Final = (
        "dezenas de milhares de dólares americanos; 1 unidade = US$ 10.000"
    )
    for dataset_key, dataset_name, evaluation_dataset in (
        ("dados_com_outliers", "Dados com outliers", dataset),
        ("dados_sem_outliers", "Dados sem outliers", dataset_without_outliers),
    ):
        print(f"\n\n######## {dataset_name} ########")
        target = evaluation_dataset.get_column(TARGET_VARIABLE_COLUMN_NAME).to_numpy()
        target_mean = float(np.mean(target))
        target_means_by_dataset[dataset_name] = target_mean
        print(
            f"Escala de {TARGET_VARIABLE_COLUMN_NAME} "
            f"({TARGET_VARIABLE_MEASUREMENT_UNIT}): "
            f"mínimo={np.min(target):.4f}; média={target_mean:.4f}; "
            f"máximo={np.max(target):.4f}"
        )
        print("RMSE e MAE estão na mesma unidade da coluna alvo.")
        linear_regression_dataset = evaluation_dataset.to_dummies(
            "ocean_proximity",
            drop_first=True,
        )
        linear_regression_feature_names = linear_regression_dataset.drop(
            TARGET_VARIABLE_COLUMN_NAME, CLUSTER_COLUMN_NAME
        ).columns
        linear_regression_scores, linear_regression_models = linear_regression_train(
            linear_regression_dataset.lazy(), target_mean
        )

        tree_based_models_dataset = evaluation_dataset.to_dummies(
            "ocean_proximity",
            drop_first=False,
        )
        tree_based_models_feature_names = tree_based_models_dataset.drop(
            TARGET_VARIABLE_COLUMN_NAME, CLUSTER_COLUMN_NAME
        ).columns
        (
            decision_tree_scores,
            decision_tree_best_hyperparameters,
            decision_tree_hyperparameter_selection,
            decision_tree_models,
        ) = decision_tree_train(tree_based_models_dataset.lazy(), target_mean)
        (
            random_forest_scores,
            random_forest_best_hyperparameters,
            random_forest_hyperparameter_selection,
            random_forest_models,
        ) = random_forest_train(tree_based_models_dataset.lazy(), target_mean)
        model_scores = [
            ("Regressão linear", linear_regression_scores),
            ("Árvore de decisão", decision_tree_scores),
            ("Floresta aleatória", random_forest_scores),
        ]
        models_comparison_print(model_scores, target_mean)
        trained_models = {
            "Regressão linear": linear_regression_models,
            "Árvore de decisão": decision_tree_models,
            "Floresta aleatória": random_forest_models,
        }
        feature_names = {
            "Regressão linear": linear_regression_feature_names,
            "Árvore de decisão": tree_based_models_feature_names,
            "Floresta aleatória": tree_based_models_feature_names,
        }
        model_feature_importances = {
            model_name: feature_importance_get(models, feature_names[model_name])
            for model_name, models in trained_models.items()
        }
        best_model_name, _ = max(model_scores, key=lambda item: np.mean(item[1]["r2"]))
        best_model_feature_importance = model_feature_importances[best_model_name]
        feature_importance_print(best_model_name, best_model_feature_importance)
        mean_scores_by_dataset[dataset_name] = {
            model_name: {
                metric: float(np.mean(metric_scores))
                for metric, metric_scores in scores.items()
            }
            for model_name, scores in model_scores
        }
        best_hyperparameters_by_dataset[dataset_name] = {
            "Árvore de decisão": decision_tree_best_hyperparameters,
            "Floresta aleatória": random_forest_best_hyperparameters,
        }
        models_results = {
            "regressao_linear": {
                "nome": "Regressão linear",
                "avaliacao": evaluation_result_get(
                    linear_regression_scores, target_mean
                ),
            },
            "arvore_de_decisao": {
                "nome": "Árvore de decisão",
                "avaliacao": evaluation_result_get(decision_tree_scores, target_mean),
                "selecao_de_hiperparametros": decision_tree_hyperparameter_selection,
            },
            "floresta_aleatoria": {
                "nome": "Floresta aleatória",
                "avaliacao": evaluation_result_get(random_forest_scores, target_mean),
                "selecao_de_hiperparametros": random_forest_hyperparameter_selection,
            },
        }
        model_result_keys = {
            "Regressão linear": "regressao_linear",
            "Árvore de decisão": "arvore_de_decisao",
            "Floresta aleatória": "floresta_aleatoria",
        }
        for model_name, model_result_key in model_result_keys.items():
            model_result = cast(dict[str, object], models_results[model_result_key])
            feature_importance = model_feature_importances[model_name]
            model_result["importancia_das_features"] = feature_importance
            model_output_path = model_bundle_save(
                dataset_key,
                dataset_name,
                model_result_key,
                model_name,
                feature_names[model_name],
                trained_models[model_name],
                cast(dict[str, object], model_result["avaliacao"]),
                feature_importance,
            )
            model_result["arquivo_dos_modelos"] = str(
                model_output_path.relative_to(EP3_DIRECTORY_PATH)
            )
            print(f"Modelos salvos em {model_output_path}")
        datasets_results[dataset_key] = {
            "nome": dataset_name,
            "quantidade_de_linhas": evaluation_dataset.height,
            "variavel_alvo": {
                "nome": TARGET_VARIABLE_COLUMN_NAME,
                "unidade_de_medida": TARGET_VARIABLE_MEASUREMENT_UNIT,
                "minimo": float(np.min(target)),
                "media": target_mean,
                "maximo": float(np.max(target)),
            },
            "modelos": models_results,
            "comparacao_dos_modelos": [
                {
                    "posicao": position,
                    "modelo": model_name,
                    "metricas_medias": dict(
                        mean_scores_by_dataset[dataset_name][model_name]
                    ),
                }
                for position, (model_name, _) in enumerate(
                    sorted(
                        model_scores,
                        key=lambda item: np.mean(item[1]["r2"]),
                        reverse=True,
                    ),
                    start=1,
                )
            ],
        }

    results["conjuntos_de_dados"] = datasets_results

    print("\n=== Sensibilidade a outliers ===")
    for model_name, scores_with_outliers in mean_scores_by_dataset[
        "Dados com outliers"
    ].items():
        scores_without_outliers = mean_scores_by_dataset["Dados sem outliers"][
            model_name
        ]
        print(f"{model_name}:")
        for metric, label in (("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")):
            score_with_outliers = scores_with_outliers[metric]
            score_without_outliers = scores_without_outliers[metric]
            normalized_difference_display = ""
            if metric == "r2":
                score_with_outliers_display = f"{score_with_outliers:.4f}"
                score_without_outliers_display = f"{score_without_outliers:.4f}"
            else:
                target_mean_with_outliers = target_means_by_dataset[
                    "Dados com outliers"
                ]
                target_mean_without_outliers = target_means_by_dataset[
                    "Dados sem outliers"
                ]
                score_with_outliers_display = error_score_format(
                    score_with_outliers,
                    target_mean_with_outliers,
                )
                score_without_outliers_display = error_score_format(
                    score_without_outliers,
                    target_mean_without_outliers,
                )
                normalized_difference = (
                    score_without_outliers / target_mean_without_outliers
                    - score_with_outliers / target_mean_with_outliers
                )
                normalized_difference_display = f""" ({normalized_difference * 100:+.2f}
                    p.p. da média da coluna alvo)"""
            print(
                f"  {label} médio com outliers={score_with_outliers_display}; "
                f"sem outliers={score_without_outliers_display}; "
                f"diferença={score_without_outliers - score_with_outliers:+.4f}"
                f"{normalized_difference_display}"
            )

    hyperparameter_comparison_print(best_hyperparameters_by_dataset)

    metric_sensitivity: dict[str, object] = {}
    for model_name, scores_with_outliers in mean_scores_by_dataset[
        "Dados com outliers"
    ].items():
        scores_without_outliers = mean_scores_by_dataset["Dados sem outliers"][
            model_name
        ]
        model_metric_sensitivity: dict[str, object] = {}
        for metric in ("r2", "rmse", "mae"):
            score_with_outliers = scores_with_outliers[metric]
            score_without_outliers = scores_without_outliers[metric]
            comparison: dict[str, object] = {
                "com_outliers": score_with_outliers,
                "sem_outliers": score_without_outliers,
                "diferenca": score_without_outliers - score_with_outliers,
            }
            if metric != "r2":
                comparison["diferenca_normalizada_pela_media_da_variavel_alvo"] = (
                    score_without_outliers
                    / target_means_by_dataset["Dados sem outliers"]
                    - score_with_outliers
                    / target_means_by_dataset["Dados com outliers"]
                )
            model_metric_sensitivity[metric] = comparison
        metric_sensitivity[model_name] = model_metric_sensitivity

    hyperparameter_sensitivity: dict[str, object] = {}
    for model_name, hyperparameters_with_outliers in best_hyperparameters_by_dataset[
        "Dados com outliers"
    ].items():
        hyperparameters_without_outliers = best_hyperparameters_by_dataset[
            "Dados sem outliers"
        ][model_name]
        hyperparameter_sensitivity[model_name] = {
            name: {
                "com_outliers": value_with_outliers,
                "sem_outliers": hyperparameters_without_outliers[name],
                "igual": value_with_outliers == hyperparameters_without_outliers[name],
            }
            for name, value_with_outliers in hyperparameters_with_outliers.items()
        }

    results["sensibilidade_a_outliers"] = {
        "metricas": metric_sensitivity,
        "hiperparametros": hyperparameter_sensitivity,
    }
    OUTPUT_DIRECTORY_PATH: Final = EP3_DIRECTORY_PATH / "output"
    OUTPUT_PATH: Final = OUTPUT_DIRECTORY_PATH / "resultados.json"
    OUTPUT_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)
    temporary_output_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_output_path.replace(OUTPUT_PATH)
    for dataset_key in datasets_results:
        legacy_model_path = (
            OUTPUT_DIRECTORY_PATH / "modelos" / f"{dataset_key}.joblib"
        )
        legacy_model_path.unlink(missing_ok=True)
    print(f"\nResultados salvos em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
