from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Create a SparkSession
spark = SparkSession.builder.appName("Bearing Data Processing").getOrCreate()

# Read JSON data from socket with the defined schema
line = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Start the streaming query to process the parsed data from the socket
query = line \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# Wait for the termination of the query
query.awaitTermination()
