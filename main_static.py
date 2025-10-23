"""
Main Script: Bronze Layer Processing
Process all 24 monthly CSV files (Jan 2023 - Dec 2024) into single Bronze Parquet

Usage:
    python main_static.py

Output:
    datamart/bronze/flight/bronze_flight_combined.parquet/
"""

import os
import time
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession

# Import our processing functions
from utils.data_processing_flight_bronze import (
    process_all_months_to_bronze,
    validate_bronze_parquet,
    print_holiday_list
)


def main():
    """
    Main execution function
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print("FLIGHT DELAY PREDICTION - BRONZE LAYER PROCESSING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Initialize Spark session
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("FlightDelayBronze") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to reduce noise
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session initialized\n")
    
    # Configuration
    data_directory = "data/flight/train/"
    bronze_output_path = "datamart/bronze/flight/bronze_flight_combined.parquet"
    
    # Check if data directory exists
    if not os.path.exists(data_directory):
        print(f"ERROR: Data directory not found: {data_directory}")
        print("\nPlease ensure your CSV files are in the correct location:")
        print(f"  {data_directory}T_ONTIME_REPORTING-01_23.csv")
        print(f"  {data_directory}T_ONTIME_REPORTING-02_23.csv")
        print("  ...")
        print(f"  {data_directory}T_ONTIME_REPORTING-12_24.csv")
        spark.stop()
        return
    
    # Print US federal holidays for review
    print_holiday_list()
    
    try:
        # Process all months to Bronze
        df_bronze = process_all_months_to_bronze(
            data_directory=data_directory,
            bronze_output_path=bronze_output_path,
            spark=spark
        )
        
        # Validate outputs
        validation_results = validate_bronze_parquet(
            bronze_output_path=bronze_output_path,
            spark=spark
        )
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"\n✓ Bronze Parquet created: {bronze_output_path}")
        print(f"✓ Total rows: {validation_results.get('total_rows', 'N/A'):,}")
        print(f"✓ Date range: {validation_results.get('min_date', 'N/A')} to {validation_results.get('max_date', 'N/A')}")
        print(f"✓ Processing time: {elapsed_time/60:.1f} minutes")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Review holiday list above to verify accuracy")
        print("2. Inspect Parquet file:")
        print(f"   ls -lh {bronze_output_path}/")
        print("\n3. Tomorrow: Build XGBoost model using this Bronze data")
        print("\n4. Later: Develop Silver/Gold layers and Airflow DAG")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print("="*80)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        print("\nStopping Spark session...")
        spark.stop()
        print("✓ Spark session stopped\n")


if __name__ == "__main__":
    main()