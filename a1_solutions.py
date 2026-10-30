# %%
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

DT_SECONDS = 1 / 20 

spark = SparkSession.builder.appName("SoccerPositionAnalysis").getOrCreate()
# %%
# Part 1
tracking_schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("tag_id", IntegerType(), True),
    StructField("x_pos", DoubleType(), True),
    StructField("y_pos", DoubleType(), True),
    StructField("heading", DoubleType(), True),
    StructField("direction", DoubleType(), True),
    StructField("energy", DoubleType(), True),
    StructField("speed", DoubleType(), True),
    StructField("total_distance", DoubleType(), True),
])

possession_schema = StructType([
    StructField("seconds", DoubleType(), True),
    StructField("team", StringType(), True)
])

def load_tracking(path, half_label):
    return (spark.read
            .option("header", "false")
            .schema(tracking_schema)
            .csv(path)
            .withColumn("half", F.lit(half_label)))

first_half  = load_tracking("2013-11-03_tromso_stromsgodset_first.csv", "first")
second_half = load_tracking("2013-11-03_tromso_stromsgodset_second.csv", "second")
tracking_df = first_half.unionByName(second_half)

poss_h1 = (spark.read
           .option("header", "false")
           .schema(possession_schema)
           .csv("g1_h1_match_state.csv")
           .withColumn("half", F.lit("first")))
poss_h2 = (spark.read
           .option("header", "false")
           .schema(possession_schema)
           .csv("g1_h2_match_state.csv")
           .withColumn("half", F.lit("second")))

possession_df = poss_h1.unionByName(poss_h2)

# Extract seconds-since-midnight from tracking timestamp string, then subtract
# the kickoff time of each half to get a comparable "seconds" value.
def add_offset_seconds(df, ts_col="timestamp", out_col="seconds"):
    frac_str = F.regexp_extract(F.col(ts_col), r"\.(\d+)$", 1)
    return (df
        .withColumn("_hh", F.substring(F.col(ts_col), 12, 2).cast("double"))
        .withColumn("_mm", F.substring(F.col(ts_col), 15, 2).cast("double"))
        .withColumn("_ss", F.substring(F.col(ts_col), 18, 2).cast("double"))
        .withColumn("_frac_str", frac_str)
        .withColumn("_frac",
            F.when(F.col("_frac_str") != "",
                   F.col("_frac_str").cast("double") / F.pow(F.lit(10), F.length("_frac_str")))
             .otherwise(F.lit(0.0)))
        .withColumn(out_col, F.col("_hh")*3600 + F.col("_mm")*60 + F.col("_ss") + F.col("_frac"))
        .drop("_hh", "_mm", "_ss", "_frac_str", "_frac"))

tracking_df = add_offset_seconds(tracking_df)

# Kickoff time = minimum timestamp offset per half - subtract to align with possession offsets
kickoff_times = (tracking_df.groupBy("half")
                  .agg(F.min("seconds").alias("kickoff_offset")))

tracking_df = (tracking_df.join(kickoff_times, on="half")
               .withColumn("seconds", F.col("seconds") - F.col("kickoff_offset"))
               .drop("kickoff_offset"))

tracking_df.cache()
possession_df.cache()

# %%
## Part 2
# 1) X/Y positions for player 6, first half
player6_first = (tracking_df
              .filter((F.col("tag_id") == 6) & (F.col("half") == "first"))
              .select("timestamp", "x_pos", "y_pos")
              .orderBy("timestamp"))
player6_pd = player6_first.toPandas()

plt.figure(figsize=(8, 5))
sc = plt.scatter(player6_pd["x_pos"], player6_pd["y_pos"],
                  c=range(len(player6_pd)), cmap="viridis", s=5)
plt.colorbar(sc, label="Time order (row index)")
plt.xlabel("X position (m)"); plt.ylabel("Y position (m)")
plt.title("Player 6 movement - first half")
plt.show()
# %%
# 2) Max speed per tag_id across both halves
max_speed_df = tracking_df.groupBy("tag_id").agg(F.max("speed").alias("max_speed")).orderBy("tag_id")
max_speed_pd = max_speed_df.toPandas()

plt.figure(figsize=(10, 5))
plt.bar(max_speed_pd["tag_id"].astype(str), max_speed_pd["max_speed"])
plt.xlabel("Tag ID"); plt.ylabel("Max speed (m/s)")
plt.title("Maximum speed per player (both halves)")
plt.xticks(rotation=90)
plt.show()

# %%
# 3) Distance between players 7 and 8, first half
p7 = (tracking_df.filter((F.col("tag_id") == 7) & (F.col("half") == "first"))
        .select("timestamp", F.col("x_pos").alias("x7"), F.col("y_pos").alias("y7")))
p8 = (tracking_df.filter((F.col("tag_id") == 8) & (F.col("half") == "first"))
    .select("timestamp", F.col("x_pos").alias("x8"), F.col("y_pos").alias("y8")))

pair_dist = (p7.join(F.broadcast(p8), on="timestamp", how="inner")
            .withColumn("distance", F.sqrt((F.col("x7")-F.col("x8"))**2 + (F.col("y7")-F.col("y8"))**2))
            .select("timestamp", "distance")
            .orderBy("timestamp"))

w = Window.orderBy("timestamp")

pair_dist_sampled = (pair_dist
                .withColumn("row_num", F.row_number().over(w))
                .filter(F.col("row_num") % 20 == 0)
                .drop("row_num"))

pair_dist_pd = pair_dist_sampled.toPandas()

row_index = np.arange(len(pair_dist_pd))
elapsed_seconds = row_index

def format_mm_ss(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

tick_interval = 300  # 5 minutes
tick_positions = np.arange(0, len(pair_dist_pd), tick_interval)
tick_labels = [format_mm_ss(pos) for pos in tick_positions]

plt.figure(figsize=(10, 5))
plt.plot(row_index, pair_dist_pd["distance"], linewidth=0.7)
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45, ha="right")
plt.xlabel("Match time (MM:SS)"); plt.ylabel("Distance (m)")
plt.title("Distance between players 7 and 8 - first half")
plt.tight_layout()
plt.show()

# %%
# 4) Team centroid and mean total_distance per time step ---
centroid_df = (tracking_df.groupBy("timestamp", "half")
                .agg(F.mean("x_pos").alias("centroid_x"),
                    F.mean("y_pos").alias("centroid_y"),
                    F.mean("total_distance").alias("mean_total_distance"))
                .orderBy("half", "timestamp"))
centroid_pd = centroid_df.toPandas()

plt.figure(figsize=(8, 5))
sc = plt.scatter(centroid_pd["centroid_x"], centroid_pd["centroid_y"],
                c=centroid_pd["mean_total_distance"], cmap="plasma", s=5)
plt.colorbar(sc, label="Mean total distance (m)")
plt.xlabel("Centroid X (m)"); plt.ylabel("Centroid Y (m)")
plt.title("Team centroid coloured by mean total distance")
plt.show()

# Define the Window
# %%
# 5) Centroid + speed for players 7 & 8, both halves
pair_positions = (tracking_df
                .filter(F.col("tag_id").isin(7, 8))
                .select("timestamp", "half", "x_pos", "y_pos"))

# Centroid = mean X/Y of players 7 and 8 at each timestamp
pair_centroid_df = (pair_positions
                    .groupBy("timestamp", "half")
                    .agg(F.mean("x_pos").alias("centroid_x"), F.mean("y_pos").alias("centroid_y"))
                    .orderBy("half", "timestamp"))

# Speed = distance moved between consecutive time steps, within each half
w = Window.partitionBy("half").orderBy("timestamp")

pair_centroid_speed_df = (pair_centroid_df
    .withColumn("prev_x", F.lag("centroid_x").over(w))
    .withColumn("prev_y", F.lag("centroid_y").over(w))
    .withColumn("dist", F.sqrt((F.col("centroid_x")-F.col("prev_x"))**2 + (F.col("centroid_y")-F.col("prev_y"))**2))
    .withColumn("centroid_speed", F.when(F.col("dist").isNotNull(), F.col("dist") / DT_SECONDS).otherwise(F.lit(0.0)))
    .select("timestamp", "half", "centroid_x", "centroid_y", "centroid_speed")
    .orderBy("half", "timestamp"))

pair_speed_pd = pair_centroid_speed_df.toPandas()

plt.figure(figsize=(8, 5))
sc = plt.scatter(pair_speed_pd["centroid_x"], pair_speed_pd["centroid_y"],
                  c=pair_speed_pd["centroid_speed"], cmap="coolwarm", s=5,
                  vmin=0, vmax=10)   # realistic human speed range in m/s
plt.colorbar(sc, label="Centroid speed (m/s)")
plt.xlabel("Centroid X (m)"); plt.ylabel("Centroid Y (m)")
plt.title("Centroid of players 7 & 8 - speed over time")
plt.show()

# %%
# Part 3: 
w_poss = Window.partitionBy("half").orderBy("seconds")

possession_intervals = (possession_df
    .withColumn("end_offset", F.lead("seconds").over(w_poss))
    .withColumnRenamed("seconds", "start_offset"))

tracking_with_poss = (tracking_df.alias("t")
    .join(possession_intervals.alias("p"),
          (F.col("t.half") == F.col("p.half")) &
          (F.col("t.seconds") >= F.col("p.start_offset")) &
          ((F.col("t.seconds") < F.col("p.end_offset")) | F.col("p.end_offset").isNull()),
          how="inner")
    .select("t.timestamp", "t.tag_id", "t.x_pos", "t.y_pos", "t.direction",
            "t.speed", "t.half", "p.team"))

tracking_with_poss.cache()
# %%
# 1) Mean speed per player when home team has possession
mean_speed_home_poss = (tracking_with_poss
    .filter(F.col("team") == "home")
    .groupBy("tag_id")
    .agg(F.mean("speed").alias("mean_speed_home_possession"))
    .orderBy("tag_id"))
mean_speed_home_poss.show(50, truncate=False)
# %%
# 2) Team centroid grouped by possession
centroid_by_poss = (tracking_with_poss
    .groupBy("timestamp", "half", "team")
    .agg(F.mean("x_pos").alias("centroid_x"), F.mean("y_pos").alias("centroid_y")))

home_centroid_pd = centroid_by_poss.filter(F.col("team") == "home").toPandas()
opp_centroid_pd  = centroid_by_poss.filter(F.col("team") != "home").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
axes[0].hist2d(home_centroid_pd["centroid_x"], home_centroid_pd["centroid_y"], bins=50, cmap="Reds")
axes[0].set_title("Home team centroid density (home possession)")
axes[1].hist2d(opp_centroid_pd["centroid_x"], opp_centroid_pd["centroid_y"], bins=50, cmap="Blues")
axes[1].set_title("Home team centroid density (opposition possession)")
for ax in axes:
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
plt.tight_layout()
plt.show()
# %%
# 3) Direction-change speed & acceleration during opposition possession
w_player = Window.partitionBy("tag_id", "half").orderBy("timestamp")

opp_possession_df = (tracking_with_poss
    .filter(F.col("team") != "home")
    .withColumn("prev_direction", F.lag("direction").over(w_player))
    .withColumn("prev_speed", F.lag("speed").over(w_player))
    .withColumn("direction_change_speed",
                F.when(F.col("prev_direction").isNotNull(),
                    (F.col("direction") - F.col("prev_direction")) / DT_SECONDS)
                .otherwise(None))
    .withColumn("acceleration",
                F.when(F.col("prev_speed").isNotNull(),
                    (F.col("speed") - F.col("prev_speed")) / DT_SECONDS)
                .otherwise(None))
    .select("timestamp", "tag_id", "direction_change_speed", "acceleration")
    .dropna())

opp_pd = opp_possession_df.toPandas()

plt.figure(figsize=(8, 6))
plt.scatter(opp_pd["direction_change_speed"], opp_pd["acceleration"], s=3, alpha=0.4)
plt.xlabel("Rate of change of direction (rad/s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Direction-change speed vs acceleration - opposition possession")
plt.show()
# %%