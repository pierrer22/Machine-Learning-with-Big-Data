# Human Activity Recognition (HAR70+) using PySpark

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Activity%20Recognition-brightgreen)

## 📌 Project Overview
This project implements a scalable machine learning pipeline in **PySpark** to classify physical activities of older adults using the **HAR70+** dataset. By applying custom rolling window feature extraction, standardization, and univariate feature selection, this repository benchmarks **Multinomial Logistic Regression**, **Decision Trees**, and **Random Forests** across multiple time-window configurations ($3\text{s}$, $5\text{s}$, and $10\text{s}$).

---

## 🛠️ Workflow Implementation & Design Choices

### 1. Dataset & Explicit Schema
The **HAR70+** dataset contains $50\text{ Hz}$ accelerometer recordings from two Axivity AX3 sensors (positioned on the lower back and right thigh) across 18 older adults, covering 7 physical activities:
* `1`: Walking
* `3`: Shuffling
* `4`: Ascending stairs
* `5`: Descending stairs
* `6`: Standing
* `7`: Sitting
* `8`: Lying down

Rather than letting Spark infer the schema—which requires extra passes over $2.26\text{ million}$ rows and risks inaccurate datatypes—an explicit `StructType` schema was defined using `DoubleType` for the six sensor axes (`back_x..z`, `thigh_x..z`), `TimestampType` for timestamps, and `IntegerType` for labels. Participant IDs were parsed directly from the CSV filenames into a separate grouping column.

### 2. Moving Window Features (`pyspark.sql.Window`)
Individual accelerometer timestamps do not provide enough context to distinguish activities like walking from shuffling. I implemented a custom PySpark Transformer (`WindowFeatureExtractor`) using:
```python
Window.partitionBy('participant_id').orderBy('timestamp').rowsBetween(-(T-1), 0)
```
Partitioning by participant ensures rolling calculations never bleed across different individuals. Across both sensors, **34 features** were extracted per window based on 7 mathematical aggregations:
1. **Mean** (per axis)
2. **Min** (per axis)
3. **Max** (per axis)
4. **Skewness** (per axis)
5. **Kurtosis** (per axis)
6. **Signal Magnitude Area (SMA):** 
   $$\text{SMA} = \frac{1}{T} \sum (|a_x| + |a_y| + |a_z|)$$
7. **Average Intensity (AI):** 
   $$\text{AI} = \frac{1}{T} \sum \sqrt{a_x^2 + a_y^2 + a_z^2}$$

These were calculated across three window sizes:
* **$3\text{s}$ Window** ($T = 150$)
* **$5\text{s}$ Window** ($T = 250$)
* **$10\text{s}$ Window** ($T = 500$)

### 3. Feature Normalisation (`StandardScaler`)
The 34 extracted features operate on vastly different scales (e.g., kurtosis can reach double digits while raw acceleration hovers around $1g$). A `StandardScaler` (`withMean=True`, `withStd=True`) stage was added to standardize features to zero mean and unit variance. This prevents high-magnitude features from dominating distance calculations and ensures fair weight penalisation during regularised Logistic Regression.

### 4. Feature Selection (ANOVA F-Test)
To remove redundant columns and accelerate training, `UnivariateFeatureSelector` was applied using an **ANOVA F-test** (continuous features, categorical label) to select the **top 12 features** from the original 34. This preserved the strongest predictors while reducing collinearity.

### 5. Subject-Wise Split & 3-Fold Cross-Validation
To evaluate how well models generalise to unseen individuals, the dataset was split by participant rather than a random row split (which would leak adjacent frames from the same walking bouts into both sets and yield artificially high scores):
* **Training Set:** Participants `501–505` ($14,902$ instances at $5\text{s}$)
* **Testing Set:** Participants `516–518` ($8,985$ instances at $5\text{s}$)

For hyperparameter tuning, PySpark's `CrossValidator` was run with 3 folds, optimising for **weighted F1-score** so smaller classes (such as stairs and shuffling) were fairly weighted. This tuned regularisation parameters for Logistic Regression, tree depth for Decision Trees, and tree counts/depth for Random Forest.

---

## 📊 Experimental Results & Performance Comparison

All three algorithms were evaluated on the held-out test participants across the $3\text{s}$, $5\text{s}$, and $10\text{s}$ window configurations. 

### Table 1: Test Performance Across Models & Windows

| Window | Algorithm | Accuracy | F1-Score (W) | Precision (W) | Recall (W) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **3 seconds** | **Logistic Regression** | **0.7361** | **0.7390** | **0.7993** | **0.7361** |
| 3 seconds | Decision Tree | 0.3000 | 0.3115 | 0.3840 | 0.3000 |
| 3 seconds | Random Forest | 0.5634 | 0.5641 | 0.6308 | 0.5634 |
| **5 seconds** | **Logistic Regression** | **0.7205** | **0.7250** | **0.7753** | **0.7205** |
| 5 seconds | Decision Tree | 0.4440 | 0.4550 | 0.5240 | 0.4440 |
| 5 seconds | Random Forest | 0.5555 | 0.5233 | 0.7177 | 0.5555 |
| **10 seconds** | **Logistic Regression** | **0.7445** | **0.7561** | **0.8010** | **0.7445** |
| 10 seconds | Decision Tree | 0.5440 | 0.5154 | 0.6494 | 0.5440 |
| 10 seconds | Random Forest | 0.5739 | 0.5559 | 0.6545 | 0.5739 |

### Visualisations
> 💡 *Note: Upload the chart images from your report to an `images/` folder in your repository to render them below.*

* **Figure 1: Benchmark Comparison Across Models & Windows**
  `![Figure 1: Benchmark Comparison](images/fig1_benchmarks.png)`
* **Figure 2: Top Random Forest Feature Importances (5s Window)**
  `![Figure 2: Feature Importances](images/fig2_feature_importance.png)`
* **Figure 3: Normalized Confusion Matrix (Random Forest - 5s Window)**
  `![Figure 3: Confusion Matrix](images/fig3_confusion_matrix.png)`
* **Figure 4: Learned Feature Space Clustering (Top 2 Features)**
  `![Figure 4: Feature Space Clustering](images/fig4_clustering.png)`

---

## 🧠 Discussion

### 1. Comparison of Algorithms
* **Multinomial Logistic Regression** proved to be the most dependable model across all window sizes, reaching $74.5\%$ accuracy, an F1-score of $0.7561$, and $80.1\%$ precision at $10\text{s}$. Because the input features were standardised and narrowed down to strong linear predictors by the ANOVA test, Logistic Regression established clean decision boundaries that generalised well to unseen test subjects.
* **Decision Tree** struggled heavily with variance, scoring only $30.0\%$ accuracy at $3\text{s}$ and $44.4\%$ at $5\text{s}$. A single tree easily overfit to the specific movement quirks of the 5 training participants.
* **Random Forest** mitigated single-tree overfitting through ensembling and feature bagging, pulling precision up to $71.8\%$ at $5\text{s}$ and producing significantly more reliable classifications.

### 2. Comparison of Window Sizes
Window length introduced a clear trade-off between statistical stability and reaction time:
* **3-Second Window ($T=150$):** Responds quickly, but $3\text{ seconds}$ is often too brief to capture complete stride cycles in older adults, leading to noisy kurtosis and skewness values that confused the tree-based models.
* **5-Second Window ($T=250$):** The optimal practical balance. It captures several full steps for walking and shuffling while still adapting quickly when a person transitions between sitting and standing.
* **10-Second Window ($T=500$):** Produced the highest raw metrics ($\text{F1} = 0.7561$) because averaging over $500$ points smooths out sensor noise. However, in real-world scenarios, a $10\text{-second}$ lag is too slow and tends to swallow brief, critical events (such as a stumble or a short flight of stairs) into surrounding sitting or standing periods.

### 3. Error Analysis & Feature Importance
* **Physics & Orientation:** Inspecting the feature importances (Figure 2), mean and minimum acceleration axes on the back and thigh contributed the most predictive power. This aligns with basic physics, as gravity acts along different sensor axes depending on whether a person is lying down, sitting, or standing upright.
* **Postures vs. Dynamic Movements:** In the confusion matrix (Figure 3), static postures (sitting, standing, lying) separated cleanly with high accuracy. Almost all classification confusion occurred between dynamic activities: *ascending vs. descending stairs*, and *shuffling vs. walking*. As shown in the 2D feature scatter (Figure 4), these moving activities generate very similar acceleration magnitudes, making them difficult to separate without a gyroscope to measure angular leg rotation.

---

## 🎯 Conclusion
Building this workflow in PySpark highlighted the critical role of feature preprocessing in human activity recognition. Utilizing `StandardScaler` and ANOVA feature selection made a major difference in helping models generalise to unseen participants. While $10\text{-second}$ windows yielded slightly higher benchmark numbers on paper, the **$5\text{-second}$ window** is the most practical choice for real-world activity monitoring, as it preserves responsiveness without obscuring rapid movement transitions.
