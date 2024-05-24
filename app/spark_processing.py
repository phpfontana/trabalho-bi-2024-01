from pyspark.sql import SparkSession
from pyspark.sql.functions import abs, input_file_name, regexp_extract, to_timestamp, date_format, mean

# Create a SparkSession
spark = SparkSession.builder.appName("Bearing Data Processing").getOrCreate()

# Load the data
df = spark.read.csv(path="data/raw/2nd_test/2nd_test", sep="\t", inferSchema=True)

# Define column names
columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

# Change the column names
for i, column_name in enumerate(columns):
    df = df.withColumnRenamed(f'_c{i}', column_name)

# Add a column with the input file name
df = df.withColumn("datetime", input_file_name())

# Extract the base name of the file from the path
filename_col = regexp_extract(df['datetime'], '([^/]+$)', 1)
df = df.withColumn("datetime", filename_col)

# Define the custom datetime format for conversion
datetime_format = "yyyy.MM.dd.HH.mm.ss"

# Convert datetime column to timestamp
df = df.withColumn("timestamp", to_timestamp("datetime", datetime_format))

# Format the timestamp column to display in the desired format
df = df.withColumn("date", date_format("timestamp", "yyyy-MM-dd HH:mm:ss"))

# Show the DataFrame with formatted date column
df.show()

# Calculate the mean of the absolute values of bearing data grouped by timestamp
df_grouped = df.groupBy("timestamp").agg(
    mean(abs(df["Bearing 1"])).alias("bearing_1"),
    mean(abs(df["Bearing 2"])).alias("bearing_2"),
    mean(abs(df["Bearing 3"])).alias("bearing_3"),
    mean(abs(df["Bearing 4"])).alias("bearing_4")
).orderBy("timestamp")

df_grouped.show(10)

# Save the DataFrame to a parquet file
df_grouped.write.csv("data/processed/2nd_test_processed.csv")

# Stop the SparkSession
spark.stop()

