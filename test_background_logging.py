#!/usr/bin/env python3

"""
Test script to verify background execution and logging functionality.
This will run a simplified version of the mean probability calculation
to test that logs are created properly.
"""

import os
import sys
import time
import logging
from datetime import datetime
import multiprocessing as mp

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the background setup and logging functions
from static_cluster_logged_mp import setup_background_process, PROCESS_LOG_DIR

def test_logging_setup():
    """Test the logging setup functionality"""
    print("Testing logging setup functionality...")
    
    # Create test log directory
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Setup a test logger similar to the mean probability function
    dev_str = "test_0.800"
    log_file = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_test.log")
    
    try:
        # Create logger
        logger = logging.getLogger(f"test_{dev_str}")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] [PID:%(process)d] [TEST] %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(file_handler)
        
        # Test logging
        logger.info("=== LOGGING TEST STARTED ===")
        logger.info(f"Log file: {log_file}")
        logger.info(f"PID: {os.getpid()}")
        logger.info("Test message 1")
        logger.info("Test message 2")
        logger.info("Test message 3")
        
        # Force flush
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        print(f"✅ Log file created: {log_file}")
        
        # Check if file exists and has content
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if content and "LOGGING TEST STARTED" in content:
                    print("✅ Log file contains expected content")
                    print(f"📄 Log file size: {len(content)} bytes")
                else:
                    print("❌ Log file is empty or missing expected content")
                    return False
        else:
            print(f"❌ Log file was not created: {log_file}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Logging setup failed: {e}")
        return False

def test_background_setup():
    """Test the background execution setup"""
    print("\nTesting background execution setup...")
    
    try:
        # Test background setup
        bg_log_file = setup_background_process()
        print(f"✅ Background setup completed")
        print(f"📄 Background log file: {bg_log_file}")
        
        # Check if background log file was created
        if os.path.exists(bg_log_file):
            with open(bg_log_file, 'r') as f:
                content = f.read()
                if content and "Background execution started" in content:
                    print("✅ Background log file contains expected content")
                else:
                    print("❌ Background log file is empty or missing expected content")
                    return False
        else:
            print(f"❌ Background log file was not created: {bg_log_file}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Background setup failed: {e}")
        return False

def test_multiprocess_logging():
    """Test logging in a multiprocess context"""
    print("\nTesting multiprocess logging...")
    
    def worker_function(worker_id):
        """Simple worker function that creates a log"""
        try:
            # Setup logging for this worker
            dev_str = f"worker_{worker_id}"
            log_file = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_test.log")
            
            logger = logging.getLogger(f"worker_{worker_id}")
            logger.setLevel(logging.INFO)
            
            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('[%(asctime)s] [PID:%(process)d] [WORKER] %(levelname)s: %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler
            logger.addHandler(file_handler)
            
            # Log some messages
            logger.info(f"=== WORKER {worker_id} STARTED ===")
            logger.info(f"PID: {os.getpid()}")
            logger.info(f"Worker ID: {worker_id}")
            
            # Simulate some work
            for i in range(5):
                logger.info(f"Processing item {i+1}/5")
                time.sleep(0.1)  # Small delay
            
            logger.info(f"=== WORKER {worker_id} COMPLETED ===")
            
            # Force flush
            for handler in logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
            
            return {"worker_id": worker_id, "log_file": log_file, "success": True}
            
        except Exception as e:
            return {"worker_id": worker_id, "log_file": "", "success": False, "error": str(e)}
    
    try:
        # Run 2 workers in parallel
        with mp.Pool(processes=2) as pool:
            results = pool.map(worker_function, [1, 2])
        
        # Check results
        all_success = True
        for result in results:
            if result["success"]:
                print(f"✅ Worker {result['worker_id']} completed successfully")
                print(f"📄 Log file: {result['log_file']}")
                
                # Check log file content
                if os.path.exists(result['log_file']):
                    with open(result['log_file'], 'r') as f:
                        content = f.read()
                        if content and f"WORKER {result['worker_id']} STARTED" in content:
                            print(f"✅ Worker {result['worker_id']} log file contains expected content")
                        else:
                            print(f"❌ Worker {result['worker_id']} log file is empty or missing content")
                            all_success = False
                else:
                    print(f"❌ Worker {result['worker_id']} log file was not created")
                    all_success = False
            else:
                print(f"❌ Worker {result['worker_id']} failed: {result.get('error', 'Unknown error')}")
                all_success = False
        
        return all_success
        
    except Exception as e:
        print(f"❌ Multiprocess logging test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing background execution and logging functionality")
    print("=" * 60)
    
    # Test 1: Basic logging setup
    test1_success = test_logging_setup()
    
    # Test 2: Background setup
    test2_success = test_background_setup()
    
    # Test 3: Multiprocess logging
    test3_success = test_multiprocess_logging()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"  Basic Logging Setup: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"  Background Setup: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"  Multiprocess Logging: {'✅ PASS' if test3_success else '❌ FAIL'}")
    
    if test1_success and test2_success and test3_success:
        print("\n🎉 All tests passed! Background execution and logging should work correctly.")
        return 0
    else:
        print("\n💥 Some tests failed! There may be issues with background execution or logging.")
        return 1

if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    mp.set_start_method('spawn', force=True)
    
    sys.exit(main())
