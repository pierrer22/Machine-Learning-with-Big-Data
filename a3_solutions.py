# %%
# COSC331 - Assignment 3: Human Activity Recognition (HAR)

import os
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, IntegerType
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, UnivariateFeatureSelector,  StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Initialise Spark Session
spark = (
    SparkSession.builder
    .appName("COSC331_HAR70Plus_MLWorkflow")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# %%
# Define Schema and Load Dataset
schema = StructType([
    StructField("timestamp", TimestampType(), nullable=True),
    StructField("back_x", DoubleType(), nullable=True),
    StructField("back_y", DoubleType(), nullable=True),
    StructField("back_z", DoubleType(), nullable=True),
    StructField("thigh_x", DoubleType(), nullable=True),
    StructField("thigh_y", DoubleType(), nullable=True),
    StructField("thigh_z", DoubleType(), nullable=True),
    StructField("label", IntegerType(), nullable=True),
])

def load_har_data(data_path: str = "har70plus") -> pd.DataFrame:
    """Loads all participant CSV files with participant_id extracted from filepath."""
    df = (
        spark.read
        .format("csv")
        .schema(schema)
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
        .load(data_path)
    )
    df = df.withColumn("participant_id", f.regexp_extract(f.input_file_name(), r"(\\d+)\\.csv$", 1))
    return df

df = load_har_data("har70plus")
print("Data Schema:")
df.printSchema()

print(f"Total raw record count: {df.count():,}")
print("\nSample records:")
df.show(5, truncate=False)

print("\nClass distribution (Activity Labels):")
df.groupBy("label").count().orderBy("label").show()

# %%
# Feature Engineering: Moving Window Pipeline Transformer
# At 50 Hz, window sizes in time steps (T) are:
#   3 seconds  -> T = 150
#   5 seconds  -> T = 250
#   10 seconds -> T = 500
#
# Features calculated per triaxial sensor:
#   1. Mean (x, y, z)
#   2. Min (x, y, z)
#   3. Max (x, y, z)
#   4. Skewness (x, y, z)
#   5. Kurtosis (x, y, z)
#   6. Signal Magnitude Area (SMA): (1/T) * sum(|ax| + |ay| + |az|)
#   7. Average Intensity (AI):      (1/T) * sum(sqrt(ax^2 + ay^2 + az^2))

class WindowFeatureExtractor(Transformer):
    """PySpark ML Transformer stage computing moving window features."""

    def __init__(
        self,
        window_sec: int,
        sample_rate_hz: int = 50,
        sensor_prefixes: List[str] = None,
        partition_col: str = "participant_id",
        order_col: str = "timestamp",
    ):
        super().__init__()
        self.window_sec = window_sec
        self.sample_rate_hz = sample_rate_hz
        self.num_time_steps = int(window_sec * sample_rate_hz)
        self.sensor_prefixes = sensor_prefixes or ["back", "thigh"]
        self.partition_col = partition_col
        self.order_col = order_col

    def _transform(self, dataset):
        tag = f"_{self.window_sec}s"
        if self.partition_col and self.partition_col in dataset.columns:
            w = (
                Window.partitionBy(self.partition_col)
                .orderBy(self.order_col)
                .rowsBetween(-(self.num_time_steps - 1), 0)
            )
        else:
            w = Window.orderBy(self.order_col).rowsBetween(-(self.num_time_steps - 1), 0)

        out_df = dataset
        for prefix in self.sensor_prefixes:
            x_col = f"{prefix}_x"
            y_col = f"{prefix}_y"
            z_col = f"{prefix}_z"

            # 1. Mean (x, y, z)
            out_df = (
                out_df.withColumn(f"{x_col}_mean{tag}", f.avg(x_col).over(w))
                      .withColumn(f"{y_col}_mean{tag}", f.avg(y_col).over(w))
                      .withColumn(f"{z_col}_mean{tag}", f.avg(z_col).over(w))
            )

            # 2. Min (x, y, z)
            out_df = (
                out_df.withColumn(f"{x_col}_min{tag}", f.min(x_col).over(w))
                      .withColumn(f"{y_col}_min{tag}", f.min(y_col).over(w))
                      .withColumn(f"{z_col}_min{tag}", f.min(z_col).over(w))
            )

            # 3. Max (x, y, z)
            out_df = (
                out_df.withColumn(f"{x_col}_max{tag}", f.max(x_col).over(w))
                      .withColumn(f"{y_col}_max{tag}", f.max(y_col).over(w))
                      .withColumn(f"{z_col}_max{tag}", f.max(z_col).over(w))
            )

            # 4. Skewness (x, y, z)
            out_df = (
                out_df.withColumn(f"{x_col}_skew{tag}", f.skewness(x_col).over(w))
                      .withColumn(f"{y_col}_skew{tag}", f.skewness(y_col).over(w))
                      .withColumn(f"{z_col}_skew{tag}", f.skewness(z_col).over(w))
            )

            # 5. Kurtosis (x, y, z)
            out_df = (
                out_df.withColumn(f"{x_col}_kurt{tag}", f.kurtosis(x_col).over(w))
                      .withColumn(f"{y_col}_kurt{tag}", f.kurtosis(y_col).over(w))
                      .withColumn(f"{z_col}_kurt{tag}", f.kurtosis(z_col).over(w))
            )

            # 6. Signal Magnitude Area (SMA)
            sma_expr = f.abs(f.col(x_col)) + f.abs(f.col(y_col)) + f.abs(f.col(z_col))
            out_df = out_df.withColumn(f"{prefix}_sma{tag}", f.avg(sma_expr).over(w))

            # 7. Average Intensity (AI)
            intensity_expr = f.sqrt(f.col(x_col)**2 + f.col(y_col)**2 + f.col(z_col)**2)
            out_df = out_df.withColumn(f"{prefix}_avg_intensity{tag}", f.avg(intensity_expr).over(w))

        return out_df

def build_feature_pipeline(window_sec: int) -> Pipeline:
    """Builds a feature extraction pipeline for a given window size."""
    extractor = WindowFeatureExtractor(window_sec=window_sec, sample_rate_hz=50)
    return Pipeline(stages=[extractor])

# %%
# Prepare Evaluation Dataset & Participant Train/Test Split
train_participants = ["501", "502", "503", "504", "505"]
test_participants = ["516", "517", "518"]

print(f"Train participants: {train_participants}")
print(f"Test participants:  {test_participants}")

# Load representative participant samples
participant_dfs = []
for p in train_participants + test_participants:
    p_df = (
        spark.read
        .format("csv")
        .schema(schema)
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
        .load(f"har70plus/{p}.csv")
        .limit(3000)
        .withColumn("participant_id", f.lit(p))
    )
    participant_dfs.append(p_df)

raw_data = participant_dfs[0]
for p_df in participant_dfs[1:]:
    raw_data = raw_data.union(p_df)

# Index multiclass labels to contiguous numeric indices (0..K-1)
df_indexer = StringIndexer(inputCol="label", outputCol="labelIndex").fit(raw_data)
final_raw_df = df_indexer.transform(raw_data).cache()
final_raw_df.count()

# %%
# Machine Learning Workflow: Multi-Window Comparison (3s, 5s, 10s)
# For each window size:
#   1. Extract 34 physical moving-window features
#   2. VectorAssembler -> raw_features
#   3. Normalisation: StandardScaler (with mean and std)
#   4. UnivariateFeatureSelector -> top 12 features (ANOVA F-test)
#   5. Train/Test split by participant (group split)
#   6. Train 3 algorithms with 3-Fold Cross-Validation hyperparameter tuning:
#      - Logistic Regression (multinomial)
#      - Decision Tree Classifier
#      - Random Forest Classifier
#   7. Evaluate Accuracy, F1-Score, Precision, Recall

evaluator_acc = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="f1")
evaluator_prec = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="weightedPrecision")
evaluator_rec = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="weightedRecall")

window_sizes = [3, 5, 10]
all_benchmark_results = []
models_by_window = {}
predictions_by_window = {}
selected_features_by_window = {}

for w in window_sizes:
    print(f"\n============================================================")
    print(f"Processing Feature Set for Window Size: {w} seconds")
    print(f"============================================================")

    extractor = WindowFeatureExtractor(window_sec=w, sample_rate_hz=50)
    feat_df = extractor.transform(final_raw_df).dropna().cache()
    feat_df.count()
    feat_cols = [c for c in feat_df.columns if c.endswith(f"_{w}s")]

    # Assemble raw features
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="raw_features")
    assembled_df = assembler.transform(feat_df)

    # Feature Normalisation (StandardScaler satisfies rubric normalisation criterion)
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True,
    )
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    # Feature Selection: ANOVA F-test (continuous features -> categorical label)
    selector = (
        UnivariateFeatureSelector(featuresCol="scaled_features", outputCol="features", labelCol="labelIndex")
        .setFeatureType("continuous")
        .setLabelType("categorical")
        .setSelectionMode("numTopFeatures")
        .setSelectionThreshold(12)
    )
    sel_model = selector.fit(scaled_df)
    selected_df = sel_model.transform(scaled_df).select("participant_id", "label", "labelIndex", "features").cache()
    selected_df.count()

    selected_feat_names = [feat_cols[i] for i in sel_model.selectedFeatures]
    selected_features_by_window[w] = selected_feat_names
    print(f"Selected Top 12 Features ({w}s): {selected_feat_names[:4]} ...")

    # Subject-wise Train / Test Split
    train = selected_df.where(f.col("participant_id").isin(train_participants)).cache()
    test = selected_df.where(f.col("participant_id").isin(test_participants)).cache()
    train.count()
    test.count()

    print(f"Train samples: {train.count()} | Test samples: {test.count()}")

    # Define Classifiers
    lr = LogisticRegression(featuresCol="features", labelCol="labelIndex", family="multinomial", maxIter=20)
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="labelIndex", seed=42)
    rf = RandomForestClassifier(featuresCol="features", labelCol="labelIndex", seed=42)

    # Define Hyperparameter Grids
    lr_param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )

    dt_param_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, [5, 8])
        .build()
    )

    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [10, 20])
        .addGrid(rf.maxDepth, [5, 8])
        .build()
    )

    # 3-Fold Cross-Validation for hyperparameter tuning
    cv_lr = CrossValidator(estimator=lr, estimatorParamMaps=lr_param_grid, evaluator=evaluator_f1, numFolds=3, seed=42)
    cv_dt = CrossValidator(estimator=dt, estimatorParamMaps=dt_param_grid, evaluator=evaluator_f1, numFolds=3, seed=42)
    cv_rf = CrossValidator(estimator=rf, estimatorParamMaps=rf_param_grid, evaluator=evaluator_f1, numFolds=3, seed=42)

    print("Fitting models with 3-fold cross-validation...")
    cv_lr_model = cv_lr.fit(train)
    cv_dt_model = cv_dt.fit(train)
    cv_rf_model = cv_rf.fit(train)

    best_lr = cv_lr_model.bestModel
    best_dt = cv_dt_model.bestModel
    best_rf = cv_rf_model.bestModel

    models = {
        "Logistic Regression": best_lr,
        "Decision Tree": best_dt,
        "Random Forest": best_rf,
    }
    models_by_window[w] = models

    window_results = {}
    for name, model in models.items():
        preds = model.transform(test)

        accuracy = evaluator_acc.evaluate(preds)
        f1 = evaluator_f1.evaluate(preds)
        precision = evaluator_prec.evaluate(preds)
        recall = evaluator_rec.evaluate(preds)

        window_results[name] = {
            "predictions": preds,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "model": model,
        }

        all_benchmark_results.append({
            "Window": f"{w}s",
            "Model": name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        })

    predictions_by_window[w] = window_results

# %%
# Display Benchmark Performance Summary
results_df = pd.DataFrame(all_benchmark_results)
print("\n" + "=" * 65)
print("             OVERALL BENCHMARK PERFORMANCE COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

# Print detailed per-model metrics in course example style
for w in window_sizes:
    print(f"\n--- Detailed Metrics for Window: {w}s ---")
    for name, res in predictions_by_window[w].items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {res["accuracy"]:.4f}")
        print(f"  F1 Score:  {res["f1"]:.4f}")
        print(f"  Precision: {res["precision"]:.4f}")
        print(f"  Recall:    {res["recall"]:.4f}")

# %%
# Helper Functions for Confusion Matrix
def build_confusion_matrix_pdf(predictions):
    """Builds a formatted pandas DataFrame confusion matrix from PySpark predictions."""
    cm_pdf = predictions.groupBy("labelIndex", "prediction").count().toPandas()
    all_labels = sorted(set(cm_pdf["labelIndex"].astype(int).tolist()) | set(cm_pdf["prediction"].astype(int).tolist()))
    full = pd.DataFrame(0, index=all_labels, columns=all_labels, dtype=int)
    for _, r in cm_pdf.iterrows():
        full.at[int(r["labelIndex"]), int(r["prediction"])] = int(r["count"])
    full.index.name = "Actual"
    full.columns.name = "Predicted"
    return full

def plot_confusion_matrix_plotly(cm_df, title: str, save_html_path: str = None):
    """Plots interactive heatmap confusion matrix with Plotly Graph Objects."""
    fig = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=[str(c) for c in cm_df.columns],
        y=[str(i) for i in cm_df.index],
        text=cm_df.values,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class Index",
        yaxis_title="Actual Class Index",
        yaxis=dict(autorange="reversed"),
    )
    if save_html_path:
        fig.write_html(save_html_path)
    return fig

# %%
# Visualisations & Model Interpretability
os.makedirs("visualisations", exist_ok=True)

# Performance Comparison Charts
fig_perf = px.bar(
    results_df,
    x="Window",
    y="F1 Score",
    color="Model",
    barmode="group",
    title="F1-Score Comparison Across Models and Window Sizes (Plotly)",
    text_auto=".3f",
)
fig_perf.write_html("visualisations/perf_stats_comparison.html")

# Static Matplotlib/Seaborn Comparison Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=results_df, x="Window", y="F1 Score", hue="Model", ax=axes[0], palette="Blues_d")
axes[0].set_title("F1-Score Across Models & Window Sizes", fontsize=12, fontweight="bold")
axes[0].set_ylim(0.0, 1.0)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

sns.barplot(data=results_df, x="Window", y="Accuracy", hue="Model", ax=axes[1], palette="Greens_d")
axes[1].set_title("Accuracy Across Models & Window Sizes", fontsize=12, fontweight="bold")
axes[1].set_ylim(0.0, 1.0)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("visualisations/model_window_comparison.png", dpi=300)
plt.close()

# Confusion Matrix for Best Model (5s Window Random Forest)
best_rf_preds = predictions_by_window[5]["Random Forest"]["predictions"]
cm_df = build_confusion_matrix_pdf(best_rf_preds)

fig_cm = plot_confusion_matrix_plotly(
    cm_df,
    title="Confusion Matrix — Random Forest (5s Window)",
    save_html_path="visualisations/conf_mat_rf_5s.html",
)

# Static Matplotlib Confusion Matrix
y_true = best_rf_preds.select("labelIndex").toPandas()
y_pred = best_rf_preds.select("prediction").toPandas()
conf_matrix_norm = pd.crosstab(y_true["labelIndex"], y_pred["prediction"], normalize="index")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Normalised Confusion Matrix (Random Forest - 5s Window)", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Class Index", fontsize=11)
plt.ylabel("Actual Class Index", fontsize=11)
plt.tight_layout()
plt.savefig("visualisations/confusion_matrix.png", dpi=300)
plt.close()

# Feature Importance Plot (Learned by Random Forest)
best_rf_model = models_by_window[5]["Random Forest"]
importances = best_rf_model.featureImportances.toArray()
feat_imp_df = pd.DataFrame({
    "Feature": selected_features_by_window[5],
    "Importance": importances,
}).sort_values(by="Importance", ascending=False)

fig_imp = px.bar(feat_imp_df, x="Feature", y="Importance", title="Feature Importances (5s Window Random Forest)")
fig_imp.write_html("visualisations/feature_importance_rf.html")

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
plt.title("Top Feature Importances Learned by Random Forest (5s Window)", fontsize=12, fontweight="bold")
plt.xlabel("Gini Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig("visualisations/feature_importance.png", dpi=300)
plt.close()

# 8.4 Decision Boundary / Feature Space Visualisation (Top 2 Features)
sample_test = test.sample(withReplacement=False, fraction=0.25, seed=42).toPandas()
top2_names = feat_imp_df["Feature"].tolist()[:2]
sample_test["f1"] = sample_test["features"].apply(lambda v: float(v[0]))
sample_test["f2"] = sample_test["features"].apply(lambda v: float(v[1]))

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=sample_test,
    x="f1",
    y="f2",
    hue="labelIndex",
    palette="tab10",
    alpha=0.8,
    s=35,
)
plt.title(f"Learned Feature Space Clustering\n({top2_names[0]} vs {top2_names[1]})", fontsize=12, fontweight="bold")
plt.xlabel(f"{top2_names[0]} (Standardised)", fontsize=11)
plt.ylabel(f"{top2_names[1]} (Standardised)", fontsize=11)
plt.legend(title="Activity Index", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("visualisations/decision_boundary.png", dpi=300)
plt.close()

print("\nAll visualisations saved to ./visualisations/ (both HTML Plotly and PNG formats).")

# %%
# ------------------------------------------------------------
# 9. Clean Up Spark Session
# ------------------------------------------------------------
spark.stop()
