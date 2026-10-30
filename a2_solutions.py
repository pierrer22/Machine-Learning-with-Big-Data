#%%
# Part 1
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructType, StructField, ByteType, IntegerType, BooleanType

spark = SparkSession.builder.appName("Diabetes_Health_Indicators").getOrCreate()

df = spark.read.csv(
    "diabetes_binary_health_indicators_BRFSS2015.csv",
    header=True,
    inferSchema=True
)

from pyspark.sql.functions import col, when

boolean_columns = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"]
integer_columns = ["BMI", "GenHlth", "MentHlth","PhysHlth", "Age", "Education", "Income"]

# Convert boolean columns to boolean type
for column in boolean_columns:
    df = df.withColumn(column, col(column).cast(BooleanType()))
# Convert integer columns to integer type
for column in integer_columns:
    df = df.withColumn(column, col(column).cast(IntegerType()))

df.show(5)
# %%
# Part 2
from pyspark.ml.feature import VectorAssembler, StandardScaler

# Assemble numeric columns
assembler = VectorAssembler(inputCols=[
                                "BMI", 
                                "MentHlth", 
                                "PhysHlth", 
                                "Age"],
                            outputCol="numeric_features")
assembled_df = assembler.transform(df)

# Scale features
scaler = StandardScaler(inputCol="numeric_features", 
                        outputCol="scaled_features", 
                        withMean=True, withStd=True)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

from pyspark.ml.feature import PCA

# Apply PCA
pca = PCA(k=4, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(scaled_df)

import plotly.express as px

# Extract variance explained by each principal component
variance_array = pca_model.explainedVariance.toArray()

# X-axis labels
component_labels = [f"PC{i+1}" for i in range(len(variance_array))]

# Build the bar chart
fig = px.bar(
    x=component_labels,
    y=variance_array,
    labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
    title='Explained Variance per Principal Component'
)
fig.show()
# %%
# Part 3
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 1. Train/Test Split & Data Balancing
# Divide the data into training and testing partitions (80/20 split)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Count the occurrences of both classes in the training data
minority_count = train_df.filter(train_df["Diabetes_binary"] == True).count()
majority_count = train_df.filter(train_df["Diabetes_binary"] == False).count()

# Calculate the fraction needed to downsample the majority class to match the minority
fraction = minority_count / majority_count

# Downsample the healthy class and combine it with the diabetic class
sampled_majority = train_df.filter(train_df["Diabetes_binary"] == False).sample(withReplacement=False, fraction=fraction, seed=42)
minority_df = train_df.filter(train_df["Diabetes_binary"] == True)
balanced_train_df = sampled_majority.union(minority_df)

# 2. Feature Engineering & Classifier Setup
feature_cols = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Convert the target variable to integer type for compatibility with the classifier
balanced_train_df = balanced_train_df.withColumn("Diabetes_binary", col("Diabetes_binary").cast("integer"))
test_df = test_df.withColumn("Diabetes_binary", col("Diabetes_binary").cast("integer"))

# Initialise the Decision Tree classifier
dt = DecisionTreeClassifier(labelCol="Diabetes_binary", featuresCol="features")

# Construct the ML Pipeline to chain the vectorizer and the classifier together
pipeline = Pipeline(stages=[assembler, dt])

# 3. Hyperparameter Tuning Setup
# Build a parameter grid to optimise the tree's max depth and max bins
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.maxBins, [32, 64]) \
    .build()

# Define the evaluator to measure success using the Area Under the ROC Curve (AUC)
evaluator = BinaryClassificationEvaluator(labelCol="Diabetes_binary", metricName="areaUnderROC")

# Setup K-fold Cross Validation to automate the tuning process on our pipeline
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3, # Tests the model in 3 distinct iterations
                    seed=42)

# 4. Training & Evaluation
# Train the model using the balanced training data
print("Training the decision tree.")
cv_model = cv.fit(balanced_train_df)

# Generate predictions on our unseen test partition
predictions = cv_model.transform(test_df)

# Calculate and present the performance metrics
auc = evaluator.evaluate(predictions)
print(f"Final Model Performance - Area Under ROC (AUC): {auc}")
#%%