import os
import re
import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import webbrowser

# ---------------------------
# CONFIGURABLE PATHS
# ---------------------------
input_file_path = "/mnt/c/Users/rubva/Documents/amazon-meta.txt"
output_file_path = "/mnt/c/Users/rubva/GitHub/WSL_Ubuntu_20.04/amazon_recommendations_output.csv"


def load_data(spark, input_file):
    """
    Loads and parses the amazon-meta.txt dataset into a Spark DataFrame for collaborative filtering.

    Args:
        spark (SparkSession): The active Spark session.
        input_file (str): Path to the dataset file.

    Returns:
        ratings_df (DataFrame): Spark DataFrame containing userId, itemId, and rating.
        asin_to_int (dict): Mapping of ASIN to integer user/item ID.
        int_to_asin (dict): Reverse mapping from integer ID to ASIN.
        asin_to_info (dict): Dictionary mapping ASIN to (title, group).
    """
    print("[INFO] Loading and processing data...")

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"[ERROR] File not found: {input_file}")

    raw_rdd = spark.sparkContext.textFile(input_file)

    asin_pattern = re.compile(r"^ASIN:\s*(\S+)")
    title_pattern = re.compile(r"^title:\s*(.*)")
    group_pattern = re.compile(r"^group:\s*(.*)")
    similar_pattern = re.compile(r"^similar:\s*\d+\s+(.*)")

    def parse_product_blocks(partition):
        block = []
        for line in partition:
            line = line.strip()
            if line.startswith("Id:"):
                if block:
                    yield block
                    block = []
            block.append(line)
        if block:
            yield block

    asin_to_info = {}
    parsed_pairs = []

    for block in raw_rdd.mapPartitions(parse_product_blocks).collect():
        asin = title = group = None
        similars = []

        for line in block:
            if line.startswith("ASIN:"):
                match = asin_pattern.match(line)
                if match:
                    asin = match.group(1)
            elif line.startswith("title:"):
                title = title_pattern.match(line).group(1)
            elif line.startswith("group:"):
                group = group_pattern.match(line).group(1)
            elif "similar:" in line:
                match = similar_pattern.match(line)
                if match:
                    similars = match.group(1).split()

        if asin:
            asin_to_info[asin] = (title or "N/A", group or "N/A")
            for sim in similars:
                parsed_pairs.append((asin, sim))

    asin_set = set()
    for u, i in parsed_pairs:
        asin_set.add(u)
        asin_set.add(i)

    asin_to_int = {asin: idx for idx, asin in enumerate(asin_set)}
    int_to_asin = {v: k for k, v in asin_to_int.items()}

    mapped_pairs = [(asin_to_int[u], asin_to_int[i]) for u, i in parsed_pairs if u in asin_to_int and i in asin_to_int]
    ratings_df = spark.createDataFrame([Row(userId=u, itemId=i, rating=1.0) for u, i in mapped_pairs])

    return ratings_df, asin_to_int, int_to_asin, asin_to_info


def train_als(ratings_df):
    """
    Trains an ALS (Alternating Least Squares) model on the ratings DataFrame.

    Args:
        ratings_df (DataFrame): Spark DataFrame containing userId, itemId, and rating.

    Returns:
        ALSModel: Trained ALS model.
    """
    print("[INFO] Training ALS model...")
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(ratings_df)
    print("[INFO] ALS model training completed.")
    return model


def generate_recommendations(model, user_count=10):
    """
    Generates top-N product recommendations for users.

    Args:
        model (ALSModel): Trained ALS model.
        user_count (int): Number of users to include in the output.

    Returns:
        DataFrame: Spark DataFrame of user recommendations.
    """
    print("[INFO] Generating recommendations...")
    user_recs = model.recommendForAllUsers(10)
    return user_recs.limit(user_count)


def save_recommendations_to_csv(recommendations_df, output_file_path, int_to_asin, asin_to_info):
    """
    Saves the top-N recommendations to a CSV file.

    Args:
        recommendations_df (DataFrame): Spark DataFrame containing user recommendations.
        output_file_path (str): Path to the output CSV file.
        int_to_asin (dict): Mapping from integer item ID to ASIN.
        asin_to_info (dict): Dictionary mapping ASIN to (title, group).
    """
    print(f"[INFO] Saving recommendations to: {output_file_path}")
    pdf = recommendations_df.toPandas()

    flattened_rows = []
    for _, row in pdf.iterrows():
        userId = row['userId']
        recs = row['recommendations']
        for rec in recs:
            itemId = rec['itemId']
            asin = int_to_asin.get(itemId, "N/A")
            title, group = asin_to_info.get(asin, ("N/A", "N/A"))
            flattened_rows.append({
                'userId': userId,
                'itemId': itemId,
                'ASIN': asin,
                'Title': title,
                'Group': group,
                'PredictedRating': rec['rating']
            })

    final_df = pd.DataFrame(flattened_rows)
    final_df.to_csv(output_file_path, index=False)
    print("[INFO] File saved successfully.")


def main():
    """
    Main execution flow:
    - Initializes Spark session
    - Loads and processes the dataset
    - Trains ALS model
    - Generates top-N recommendations
    - Saves the recommendations to CSV
    - Opens Spark Web UI for visualization
    """
    spark = None
    try:
        spark = SparkSession.builder \
            .appName("AmazonRecommendationSystem") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")
        print("[INFO] Spark session started successfully.")

        web_ui_url = "http://localhost:4040"
        print(f"[INFO] Spark Web UI is available at: {web_ui_url}")
        webbrowser.open(web_ui_url)

        ratings_df, asin_to_int, int_to_asin, asin_to_info = load_data(spark, input_file_path)
        model = train_als(ratings_df)
        recommendations_df = generate_recommendations(model)
        save_recommendations_to_csv(recommendations_df, output_file_path, int_to_asin, asin_to_info)

        print("\n[INFO] Processing complete.")
        input("[INFO] Press ENTER to close the Spark session and exit...")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if spark:
            try:
                spark.stop()
                print("[INFO] Spark session stopped.")
            except Exception as e:
                print(f"[ERROR] Failed to stop Spark session: {e}")


if __name__ == "__main__":
    main()
