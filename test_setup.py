"""
Environment Setup Validation Script

Tests:
1. Python environment
2. Required packages
3. Spark initialization
4. Data file availability
5. Sample data loading

Usage:
    python test_setup.py
"""

import os
import sys
from datetime import datetime


def test_python_version():
    """Test Python version"""
    print("\n1. Testing Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("   ✓ PASS - Python 3.10+ detected")
        return True
    else:
        print("   ✗ FAIL - Python 3.10+ required")
        return False


def test_package_imports():
    """Test required package imports"""
    print("\n2. Testing package imports...")
    
    packages = {
        'pyspark': 'pyspark',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'holidays': 'holidays',
        'pyarrow': 'pyarrow'
    }
    
    all_passed = True
    
    for name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - NOT INSTALLED")
            all_passed = False
    
    if all_passed:
        print("   ✓ PASS - All packages available")
    else:
        print("   ✗ FAIL - Missing packages (run: pip install -r requirements.txt)")
    
    return all_passed


def test_spark_initialization():
    """Test Spark session creation"""
    print("\n3. Testing Spark initialization...")
    
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName("test") \
            .master("local[*]") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        
        print("   ✓ Spark session created")
        
        # Test basic operation
        test_data = [(1, "test"), (2, "data")]
        df = spark.createDataFrame(test_data, ["id", "value"])
        count = df.count()
        
        print(f"   ✓ Basic DataFrame operations working (test count: {count})")
        
        spark.stop()
        print("   ✓ PASS - Spark initialization successful")
        return True
        
    except Exception as e:
        print(f"   ✗ FAIL - Spark error: {str(e)}")
        return False


def test_data_directory():
    """Test data directory structure"""
    print("\n4. Testing data directory...")
    
    data_dir = "data/flight/train/"
    
    if not os.path.exists("data"):
        print(f"   ✗ FAIL - 'data/' directory not found")
        print("\n   Please create directory structure:")
        print("   data/")
        print("   └── flight/")
        print("       └── train/")
        print("           ├── T_ONTIME_REPORTING-01_23.csv")
        print("           ├── T_ONTIME_REPORTING-02_23.csv")
        print("           └── ...")
        return False
    
    if not os.path.exists(data_dir):
        print(f"   ✗ FAIL - '{data_dir}' directory not found")
        return False
    
    print(f"   ✓ Data directory exists: {data_dir}")
    
    # Check for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print(f"   Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("   ✗ FAIL - No CSV files found")
        print("\n   Please add CSV files to data/flight/train/")
        return False
    
    if len(csv_files) < 24:
        print(f"   ⚠ WARNING - Expected 24 CSV files, found {len(csv_files)}")
        print("   Missing months may cause processing errors")
    
    # Show first few files
    print("\n   Sample files found:")
    for f in sorted(csv_files)[:5]:
        print(f"   - {f}")
    
    if len(csv_files) > 5:
        print(f"   ... and {len(csv_files) - 5} more")
    
    if len(csv_files) >= 24:
        print("   ✓ PASS - All 24 CSV files present")
        return True
    else:
        print(f"   ⚠ PARTIAL - {len(csv_files)}/24 files present")
        return True  # Allow partial for testing


def test_sample_data_loading():
    """Test loading a sample CSV file"""
    print("\n5. Testing sample data loading...")
    
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName("test_load") \
            .master("local[*]") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        
        # Find first CSV file
        data_dir = "data/flight/train/"
        csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        
        if not csv_files:
            print("   ✗ FAIL - No CSV files to test")
            spark.stop()
            return False
        
        test_file = os.path.join(data_dir, csv_files[0])
        print(f"   Loading: {csv_files[0]}")
        
        df = spark.read.csv(test_file, header=True, inferSchema=True)
        
        row_count = df.count()
        col_count = len(df.columns)
        
        print(f"   ✓ Loaded successfully")
        print(f"   Rows: {row_count:,}")
        print(f"   Columns: {col_count}")
        
        # Check for required columns
        required_cols = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'ORIGIN', 'DEST', 
                        'ARR_DELAY_NEW', 'CRS_DEP_TIME', 'CRS_ARR_TIME']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ✗ FAIL - Missing required columns: {missing_cols}")
            spark.stop()
            return False
        
        print("   ✓ All required columns present")
        
        # Check for NYC airports
        nyc_airports = ['JFK', 'LGA', 'EWR']
        from pyspark.sql.functions import col
        
        nyc_flights = df.filter(
            col('ORIGIN').isin(nyc_airports) | col('DEST').isin(nyc_airports)
        ).count()
        
        if nyc_flights > 0:
            print(f"   ✓ NYC metro flights found: {nyc_flights:,} ({nyc_flights/row_count*100:.1f}%)")
        else:
            print("   ⚠ WARNING - No NYC metro flights found in sample")
        
        # Show sample data
        print("\n   Sample records (first 3):")
        df.select('YEAR', 'MONTH', 'DAY_OF_MONTH', 'ORIGIN', 'DEST', 'ARR_DELAY_NEW') \
            .show(3, truncate=False)
        
        spark.stop()
        print("   ✓ PASS - Sample data loading successful")
        return True
        
    except Exception as e:
        print(f"   ✗ FAIL - Error loading data: {str(e)}")
        return False


def test_output_directory():
    """Test output directory creation"""
    print("\n6. Testing output directory...")
    
    output_dir = "datamart/bronze/flight/"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"   ✓ Output directory ready: {output_dir}")
        
        # Test write permissions
        test_file = os.path.join(output_dir, ".test_write")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        print("   ✓ Write permissions confirmed")
        print("   ✓ PASS - Output directory accessible")
        return True
        
    except Exception as e:
        print(f"   ✗ FAIL - Cannot create/write to output directory: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("ENVIRONMENT SETUP VALIDATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        test_python_version,
        test_package_imports,
        test_spark_initialization,
        test_data_directory,
        test_sample_data_loading,
        test_output_directory
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n   ✗ EXCEPTION: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nEnvironment is ready! You can now run:")
        print("  python main_static.py")
    elif passed >= total - 1:
        print("\n⚠ MOSTLY READY - One test failed")
        print("Review errors above and fix before running main_static.py")
    else:
        print("\n✗✗✗ SETUP INCOMPLETE ✗✗✗")
        print("Please fix the issues above before proceeding")
    
    print("\n" + "="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)