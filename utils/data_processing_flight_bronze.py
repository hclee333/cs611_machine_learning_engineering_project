"""
Bronze Layer Processing for Flight Delay Data

Processes raw flight CSV files into a single combined Parquet file with:
- NYC metro airport filtering (JFK, LGA, EWR)
- Temporal features (DayOfWeek, IsWeekend, IsPublicHoliday)
- Target variable (is_delayed_15)
- Proper sorting and partitioning
"""

import os
from datetime import datetime
from typing import List, Dict
import holidays

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, when, to_date, concat_ws, current_timestamp,
    dayofweek, udf, coalesce
)
from pyspark.sql.types import IntegerType, DateType, StringType
from pyspark.sql.window import Window


# NYC Metro airports - our scope
NYC_METRO_AIRPORTS = ['JFK', 'LGA', 'EWR']

# US Federal holidays for our date range
US_HOLIDAYS = holidays.US(years=range(2023, 2026))


def get_csv_filename(year: int, month: int) -> str:
    """
    Map year/month to CSV filename format
    
    Args:
        year: Year (e.g., 2023)
        month: Month (1-12)
    
    Returns:
        Filename like 'T_ONTIME_REPORTING-01_23.csv'
    """
    month_str = f"{month:02d}"
    year_str = str(year)[-2:]  # Last 2 digits
    return f"T_ONTIME_REPORTING-{month_str}_{year_str}.csv"


def process_single_csv(csv_path: str, spark: SparkSession) -> DataFrame:
    """
    Load and process a single monthly CSV file
    
    Args:
        csv_path: Path to CSV file
        spark: SparkSession
    
    Returns:
        Processed DataFrame
    """
    print(f"  Loading: {os.path.basename(csv_path)}")
    
    # Read CSV
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    
    initial_count = df.count()
    print(f"    Initial rows: {initial_count:,}")
    
    # Derive FlightDate from year/month/day columns
    df = df.withColumn(
        'FlightDate',
        to_date(
            concat_ws('-', 
                     col('YEAR').cast(StringType()),
                     col('MONTH').cast(StringType()),
                     col('DAY_OF_MONTH').cast(StringType())
            ),
            'yyyy-M-d'
        )
    )
    
    # Filter for NYC metro airports only
    df = df.filter(
        col('ORIGIN').isin(NYC_METRO_AIRPORTS) | 
        col('DEST').isin(NYC_METRO_AIRPORTS)
    )
    
    filtered_count = df.count()
    print(f"    After NYC filter: {filtered_count:,} ({filtered_count/initial_count*100:.1f}%)")
    
    # Add source file metadata
    df = df.withColumn('source_file', lit(os.path.basename(csv_path)))
    
    return df


def add_derived_columns(df: DataFrame) -> DataFrame:
    """
    Add all derived columns to DataFrame
    
    Derived columns:
    - year_month: For partitioning
    - sort_time: For sorting (departure or arrival time based on direction)
    - is_delayed_15: Target variable
    - DayOfWeek: 1=Sunday, 7=Saturday
    - IsWeekend: 1 if weekend, 0 otherwise
    - IsPublicHoliday: 1 if US federal holiday, 0 otherwise
    - processing_timestamp: When this record was processed
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with derived columns
    """
    print("\n  Adding derived columns...")
    
    # 1. year_month for partitioning
    df = df.withColumn(
        'year_month',
        concat_ws('-', 
                 col('YEAR').cast(StringType()),
                 when(col('MONTH') < 10, concat_ws('', lit('0'), col('MONTH').cast(StringType())))
                 .otherwise(col('MONTH').cast(StringType()))
        )
    )
    
    # 2. sort_time - use departure time if departing FROM NYC, else arrival time
    df = df.withColumn(
        'sort_time',
        when(col('ORIGIN').isin(NYC_METRO_AIRPORTS), col('CRS_DEP_TIME'))
        .otherwise(col('CRS_ARR_TIME'))
    )
    
    # 3. is_delayed_15 - target variable (1 if arrival delay >= 15 mins)
    # Handle NULLs (cancelled/diverted flights) as 0
    df = df.withColumn(
        'is_delayed_15',
        when(col('ARR_DELAY_NEW') >= 15, 1)
        .otherwise(0)
    )
    
    # 4. DayOfWeek (1=Sunday, 2=Monday, ..., 7=Saturday)
    df = df.withColumn('DayOfWeek', dayofweek(col('FlightDate')))
    
    # 5. IsWeekend (1 if Saturday or Sunday)
    df = df.withColumn(
        'IsWeekend',
        when(col('DayOfWeek').isin([1, 7]), 1).otherwise(0)
    )
    
    # 6. IsPublicHoliday using holidays library
    # Create UDF to check if date is a US federal holiday
    @udf(returnType=IntegerType())
    def is_holiday(date):
        if date is None:
            return 0
        return 1 if date in US_HOLIDAYS else 0
    
    df = df.withColumn('IsPublicHoliday', is_holiday(col('FlightDate')))
    
    # 7. processing_timestamp
    df = df.withColumn('processing_timestamp', current_timestamp())
    
    print("    ✓ year_month")
    print("    ✓ sort_time")
    print("    ✓ is_delayed_15")
    print("    ✓ DayOfWeek")
    print("    ✓ IsWeekend")
    print("    ✓ IsPublicHoliday")
    print("    ✓ processing_timestamp")
    
    return df


def generate_monthly_dates(start_year: int, start_month: int, 
                          end_year: int, end_month: int) -> List[tuple]:
    """
    Generate list of (year, month) tuples for date range
    
    Args:
        start_year: Starting year
        start_month: Starting month (1-12)
        end_year: Ending year
        end_month: Ending month (1-12)
    
    Returns:
        List of (year, month) tuples
    """
    dates = []
    current_year = start_year
    current_month = start_month
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        dates.append((current_year, current_month))
        
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    return dates


def process_all_months_to_bronze(data_directory: str, 
                                 bronze_output_path: str,
                                 spark: SparkSession) -> DataFrame:
    """
    Main processing function: Combine all monthly CSVs into single Bronze Parquet
    
    Processing steps:
    1. Load all 24 monthly CSV files
    2. Filter for NYC metro airports (JFK, LGA, EWR)
    3. Add derived columns (target, temporal features, etc.)
    4. Union all DataFrames
    5. Sort by FlightDate and sort_time
    6. Save as partitioned Parquet
    
    Args:
        data_directory: Directory containing CSV files (e.g., 'data/flight/train/')
        bronze_output_path: Output path for Parquet (e.g., 'datamart/bronze/flight/bronze_flight_combined.parquet')
        spark: SparkSession
    
    Returns:
        Final combined DataFrame
    """
    print("="*80)
    print("BRONZE LAYER PROCESSING - FLIGHT DELAY DATA")
    print("="*80)
    
    # Generate list of months to process (Jan 2023 - Dec 2024)
    monthly_dates = generate_monthly_dates(2023, 1, 2024, 12)
    print(f"\nProcessing {len(monthly_dates)} months: Jan 2023 - Dec 2024")
    
    # Process each month
    dfs = []
    for year, month in monthly_dates:
        csv_filename = get_csv_filename(year, month)
        csv_path = os.path.join(data_directory, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"\n  WARNING: File not found: {csv_filename}")
            continue
        
        df_month = process_single_csv(csv_path, spark)
        dfs.append(df_month)
    
    if not dfs:
        raise ValueError("No CSV files were successfully processed!")
    
    print(f"\n{'='*80}")
    print("COMBINING DATA")
    print("="*80)
    
    # Union all monthly DataFrames
    print("\n  Combining all months...")
    df_combined = dfs[0]
    for df in dfs[1:]:
        df_combined = df_combined.union(df)
    
    total_rows = df_combined.count()
    print(f"  Total rows after union: {total_rows:,}")
    
    # Add derived columns
    df_combined = add_derived_columns(df_combined)
    
    # Sort by FlightDate and sort_time
    print("\n  Sorting by FlightDate and sort_time...")
    df_combined = df_combined.orderBy(['FlightDate', 'sort_time'])
    
    print(f"\n{'='*80}")
    print("SAVING TO PARQUET")
    print("="*80)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(bronze_output_path), exist_ok=True)
    
    # Save as Parquet partitioned by year_month
    print(f"\n  Output path: {bronze_output_path}")
    print("  Partitioning by: year_month")
    print("  Compression: Snappy (default)")
    print("\n  Writing Parquet file...")
    
    df_combined.write.mode('overwrite').partitionBy('year_month').parquet(bronze_output_path)
    
    print("  ✓ Parquet file saved successfully!")
    
    return df_combined


def validate_bronze_parquet(bronze_output_path: str, spark: SparkSession) -> Dict:
    """
    Validate Bronze Parquet file quality
    
    Checks:
    - Row count
    - Date coverage
    - NYC metro filter coverage
    - Target variable distribution
    - Schema validation
    
    Args:
        bronze_output_path: Path to Bronze Parquet
        spark: SparkSession
    
    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*80}")
    print("VALIDATION CHECKS")
    print("="*80)
    
    # Read Parquet
    df = spark.read.parquet(bronze_output_path)
    
    validation_results = {}
    
    # 1. Row count
    total_rows = df.count()
    validation_results['total_rows'] = total_rows
    print(f"\n1. Total Rows: {total_rows:,}")
    
    if 900000 <= total_rows <= 1200000:
        print("   ✓ PASS - Row count within expected range")
    else:
        print("   ⚠ WARNING - Row count outside expected range (900K-1.2M)")
    
    # 2. Date coverage
    date_stats = df.agg({'FlightDate': 'min', 'FlightDate': 'max'}).collect()[0]
    min_date = df.agg({'FlightDate': 'min'}).collect()[0][0]
    max_date = df.agg({'FlightDate': 'max'}).collect()[0][0]
    
    validation_results['min_date'] = str(min_date)
    validation_results['max_date'] = str(max_date)
    
    print(f"\n2. Date Range: {min_date} to {max_date}")
    
    if str(min_date).startswith('2023-01') and str(max_date).startswith('2024-12'):
        print("   ✓ PASS - Date range covers Jan 2023 to Dec 2024")
    else:
        print("   ⚠ WARNING - Unexpected date range")
    
    # 3. Month coverage
    unique_months = df.select('year_month').distinct().count()
    validation_results['unique_months'] = unique_months
    
    print(f"\n3. Month Coverage: {unique_months} unique months")
    
    if unique_months == 24:
        print("   ✓ PASS - All 24 months present")
    else:
        print(f"   ⚠ WARNING - Expected 24 months, found {unique_months}")
        missing_months = df.groupBy('year_month').count().orderBy('year_month')
        print("\n   Month distribution:")
        missing_months.show(24, truncate=False)
    
    # 4. NYC metro filter validation
    print("\n4. NYC Metro Filter Validation:")
    
    nyc_origin = df.filter(col('ORIGIN').isin(NYC_METRO_AIRPORTS)).count()
    nyc_dest = df.filter(col('DEST').isin(NYC_METRO_AIRPORTS)).count()
    nyc_total = df.filter(
        col('ORIGIN').isin(NYC_METRO_AIRPORTS) | col('DEST').isin(NYC_METRO_AIRPORTS)
    ).count()
    
    validation_results['nyc_coverage'] = nyc_total
    
    print(f"   Flights FROM NYC metro: {nyc_origin:,}")
    print(f"   Flights TO NYC metro: {nyc_dest:,}")
    print(f"   Total NYC flights: {nyc_total:,}")
    
    if nyc_total == total_rows:
        print("   ✓ PASS - 100% of flights involve NYC metro airports")
    else:
        print(f"   ✗ FAIL - Only {nyc_total/total_rows*100:.1f}% coverage")
    
    # 5. Airport distribution
    print("\n5. Airport Distribution:")
    airport_dist = df.groupBy('ORIGIN').count().orderBy(col('count').desc())
    print("\n   Top Origin Airports:")
    airport_dist.show(10)
    
    # 6. Target variable distribution
    print("\n6. Target Variable (is_delayed_15):")
    delay_dist = df.groupBy('is_delayed_15').count().collect()
    
    for row in delay_dist:
        delay_val = row['is_delayed_15']
        count = row['count']
        pct = count / total_rows * 100
        label = "Delayed (≥15 min)" if delay_val == 1 else "On-time (<15 min)"
        print(f"   {label}: {count:,} ({pct:.2f}%)")
        validation_results[f'delay_{delay_val}'] = count
    
    # 7. Cancelled flights
    cancelled_count = df.filter(col('CANCELLED') == 1).count()
    validation_results['cancelled'] = cancelled_count
    print(f"\n7. Cancelled Flights: {cancelled_count:,} ({cancelled_count/total_rows*100:.2f}%)")
    
    # 8. Holiday detection
    holiday_count = df.filter(col('IsPublicHoliday') == 1).count()
    validation_results['holiday_flights'] = holiday_count
    print(f"\n8. Flights on Public Holidays: {holiday_count:,}")
    
    # Show sample of holidays
    print("\n   Sample holiday dates:")
    df.filter(col('IsPublicHoliday') == 1).select('FlightDate').distinct() \
        .orderBy('FlightDate').show(10, truncate=False)
    
    # 9. Schema check
    print("\n9. Schema:")
    print(f"   Total columns: {len(df.columns)}")
    
    expected_derived_cols = ['year_month', 'sort_time', 'is_delayed_15', 
                             'DayOfWeek', 'IsWeekend', 'IsPublicHoliday',
                             'source_file', 'processing_timestamp', 'FlightDate']
    
    missing_cols = [col for col in expected_derived_cols if col not in df.columns]
    
    if not missing_cols:
        print("   ✓ PASS - All derived columns present")
    else:
        print(f"   ✗ FAIL - Missing columns: {missing_cols}")
    
    validation_results['schema_valid'] = len(missing_cols) == 0
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print("="*80)
    
    return validation_results


def print_holiday_list():
    """
    Print list of US federal holidays detected for manual review
    """
    print(f"\n{'='*80}")
    print("US FEDERAL HOLIDAYS (2023-2025)")
    print("="*80)
    
    for year in range(2023, 2026):
        print(f"\n{year}:")
        year_holidays = sorted([(date, name) for date, name in US_HOLIDAYS.items() 
                                if date.year == year])
        for date, name in year_holidays:
            print(f"  {date} - {name}")
    
    print(f"\n{'='*80}\n")