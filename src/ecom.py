# ==========================================
# E-Commerce Big Data Analytics & Recommender
# ==========================================

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, datediff, explode, count, date_format, when
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# --- 1. Environment Setup & Directory Management ---
# Create necessary directories for a professional project structure
os.makedirs("./data/raw", exist_ok=True)
os.makedirs("./data/processed", exist_ok=True)
os.makedirs("./output/charts", exist_ok=True)

spark = SparkSession.builder \
    .appName("Ecommerce_BigData_Project") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# --- 2. Data Loading ---
print("Loading datasets...")
# Note: Ensure all CSV files are placed in the ./data/raw/ folder
orders_df = spark.read.csv("./data/raw/olist_orders_dataset.csv", header=True, inferSchema=True)
items_df = spark.read.csv("./data/raw/olist_order_items_dataset.csv", header=True, inferSchema=True)
customers_df = spark.read.csv("./data/raw/olist_customers_dataset.csv", header=True, inferSchema=True)
reviews_df = spark.read.csv("./data/raw/olist_order_reviews_dataset.csv", header=True, inferSchema=True)
products_df = spark.read.csv("./data/raw/olist_products_dataset.csv", header=True, inferSchema=True)
translations_df = spark.read.csv("./data/raw/product_category_name_translation.csv", header=True, inferSchema=True)

# --- 3. Data Preprocessing & Cleaning ---
print("Preprocessing and cleaning data...")
# Convert date strings to timestamps
orders_df = orders_df.withColumn("order_purchase_timestamp", to_timestamp(col("order_purchase_timestamp"))) \
                     .withColumn("order_delivered_customer_date", to_timestamp(col("order_delivered_customer_date"))) \
                     .withColumn("order_estimated_delivery_date", to_timestamp(col("order_estimated_delivery_date")))

# Join Tables & Translate Portuguese Categories to English
master_df = orders_df.join(customers_df, "customer_id") \
                     .join(reviews_df, "order_id") \
                     .join(items_df, "order_id") \
                     .join(products_df, "product_id") \
                     .join(translations_df, "product_category_name", "left")

# Data Cleaning: Drop rows with missing critical values
master_df = master_df.dropna(subset=["review_score", "price", "customer_state", "product_category_name_english"])

# Save the cleaned, processed data as Parquet (Big Data best practice)
print("Saving processed data to Parquet format...")
master_df.write.mode("overwrite").parquet("./data/processed/master_dataframe")
master_df.cache()

# --- 4. Part A: Comprehensive Big Data Analytics (EDA) ---
print("Starting Part A: Advanced Analytics & Generating Charts...")

# Feature Engineering
analytics_df = master_df.withColumn("delivery_days", datediff(col("order_delivered_customer_date"), col("order_purchase_timestamp"))) \
                        .withColumn("is_late", when(col("order_delivered_customer_date") > col("order_estimated_delivery_date"), 1).otherwise(0)) \
                        .withColumn("year_month", date_format(col("order_purchase_timestamp"), "yyyy-MM"))

# Chart 1: Top 10 Product Categories by Revenue
category_revenue = master_df.groupBy("product_category_name_english").sum("price").withColumnRenamed("sum(price)", "total_revenue").orderBy("total_revenue", ascending=False).limit(10)
pdf_category = category_revenue.toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(x='total_revenue', y='product_category_name_english', data=pdf_category, palette='viridis', hue='product_category_name_english', legend=False)
plt.title('Chart 1: Top 10 Product Categories by Revenue')
plt.xlabel('Total Revenue (BRL)')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig("./output/charts/1_top_categories.png", dpi=300)
plt.close()

# Chart 2: Order Volume Time-Series Trend
monthly_orders = analytics_df.groupBy("year_month").count().orderBy("year_month")
pdf_trend = monthly_orders.toPandas()
pdf_trend = pdf_trend[(pdf_trend['year_month'] >= '2017-01') & (pdf_trend['year_month'] <= '2018-08')] # Filter edge dates

plt.figure(figsize=(12, 6))
plt.plot(pdf_trend['year_month'], pdf_trend['count'], marker='o', linestyle='-', color='#d32f2f', linewidth=2)
plt.xticks(rotation=45)
plt.title('Chart 2: Monthly Order Volume Trends (2017-2018)')
plt.xlabel('Year-Month')
plt.ylabel('Total Orders')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("./output/charts/2_order_trends.png", dpi=300)
plt.close()

# Chart 3: Logistics - Late Delivery Rate by State
state_performance = analytics_df.groupBy("customer_state").agg({"is_late": "mean"}).withColumnRenamed("avg(is_late)", "late_delivery_rate").orderBy("late_delivery_rate", ascending=False).limit(10)
pdf_state = state_performance.toPandas()
pdf_state['late_delivery_rate'] = pdf_state['late_delivery_rate'] * 100 # Convert to percentage

plt.figure(figsize=(10, 6))
sns.barplot(x='customer_state', y='late_delivery_rate', data=pdf_state, palette='magma', hue='customer_state', legend=False)
plt.title('Chart 3: Top 10 States with Highest Late Delivery Rates')
plt.xlabel('State Code')
plt.ylabel('Late Delivery Rate (%)')
plt.tight_layout()
plt.savefig("./output/charts/3_late_deliveries.png", dpi=300)
plt.close()

# Chart 4: Customer Satisfaction (Review Scores)
review_dist = master_df.groupBy("review_score").count().orderBy("review_score")
pdf_reviews = review_dist.toPandas()

plt.figure(figsize=(8, 6))
sns.barplot(x='review_score', y='count', data=pdf_reviews, palette='coolwarm', hue='review_score', legend=False)
plt.title('Chart 4: Distribution of Customer Review Scores')
plt.xlabel('Review Score (1 to 5)')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig("./output/charts/4_review_scores.png", dpi=300)
plt.close()
print("All 4 charts saved successfully to ./output/charts/")

# --- 5. Part B: Recommendation System (ALS) ---
print("Starting Part B: Recommendation System...")

# Filter for sparsity: Only keep users who have reviewed at least 2 items
user_counts = master_df.groupBy("customer_unique_id").count().filter(col("count") >= 2)
active_users_df = master_df.join(user_counts, "customer_unique_id")

# Convert IDs to Numeric Indices for Spark MLlib
user_indexer = StringIndexer(inputCol="customer_unique_id", outputCol="user_index").fit(active_users_df)
product_indexer = StringIndexer(inputCol="product_id", outputCol="product_index").fit(active_users_df)

indexed_df = user_indexer.transform(active_users_df)
indexed_df = product_indexer.transform(indexed_df)

recommender_data = indexed_df.select(
    col("user_index").cast("integer"), 
    col("product_index").cast("integer"), 
    col("review_score").cast("float")
)

(training, test) = recommender_data.randomSplit([0.8, 0.2], seed=42)

# Configure and Train ALS
als = ALS(maxIter=10, regParam=0.1, userCol="user_index", itemCol="product_index", ratingCol="review_score", coldStartStrategy="drop", nonnegative=True)
model = als.fit(training)

# Evaluate the Model (RMSE)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="review_score", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Collaborative Filtering Model RMSE = {rmse:.4f}")

# Generate Top 5 Recommendations for all users
userRecs = model.recommendForAllUsers(5)

# Flatten the recommendations array
flat_recs = userRecs.withColumn("rec", explode(col("recommendations"))) \
                    .select(col("user_index"), 
                            col("rec.product_index").alias("product_index"), 
                            col("rec.rating").alias("predicted_rating"))

# CRUCIAL: Map the predicted product_index back to the actual product_id
converter = IndexToString(inputCol="product_index", outputCol="original_product_id", labels=product_indexer.labels)
final_recs = converter.transform(flat_recs)

# Join with translations to show readable product names to the user
readable_recs = final_recs.join(products_df.select("product_id", "product_category_name"), final_recs.original_product_id == products_df.product_id, "left") \
                          .join(translations_df, "product_category_name", "left") \
                          .select("user_index", "original_product_id", "product_category_name_english", "predicted_rating") \
                          .orderBy("user_index", "predicted_rating", ascending=[True, False])

print("Sample of Actual Product Recommendations for Users:")
readable_recs.show(10, truncate=False)

# Stop Spark session
spark.stop()