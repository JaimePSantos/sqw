#!/usr/bin/env python3

"""
Demo script showing the improved logging integration for archiving.
This shows how archiving operations are now logged to both console and the multiprocess log file.
"""

import logging
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_logging():
    print("=== DEMO: ARCHIVING WITH MULTIPROCESS LOGGING ===\n")
    
    # Setup a demo logger similar to the master logger
    demo_logger = logging.getLogger("demo_master")
    demo_logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] [DEMO_MASTER] %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    demo_logger.addHandler(console_handler)
    
    # Create file handler to show log file output
    log_filename = "demo_archive_logging.log"
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s] [DEMO_MASTER] %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    demo_logger.addHandler(file_handler)
    
    print("📝 Created demo logger with both console and file output")
    print(f"📁 Log file: {log_filename}")
    print()
    
    # Import the archiving function
    try:
        from static_cluster_logged_mp import create_experiment_archive
        print("✅ Successfully imported archiving function with logging support")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return
    
    print("\n🔧 Demonstrating archiving with logging integration...")
    print("Note: This will show [WARNING] since no experiment data exists locally")
    print("But you can see how the logging works!\n")
    
    # Call the archiving function with the demo logger
    demo_logger.info("Starting archiving demonstration")
    
    result = create_experiment_archive(
        N=15000, 
        samples=5, 
        use_multiprocess=True, 
        max_archive_processes=2, 
        logger=demo_logger
    )
    
    if result:
        demo_logger.info(f"Archiving completed successfully: {result}")
    else:
        demo_logger.warning("Archiving was skipped (expected - no data present)")
    
    demo_logger.info("Archiving demonstration completed")
    
    print(f"\n📋 Check the log file '{log_filename}' to see how archiving operations are logged!")
    print("This is exactly how it will appear in your multiprocess log file.")
    
    # Show a sample of what was logged
    print(f"\n📖 Sample of what was logged to '{log_filename}':")
    print("-" * 60)
    try:
        with open(log_filename, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:  # Show last 5 lines
                print(line.strip())
    except Exception as e:
        print(f"Could not read log file: {e}")
    print("-" * 60)
    
    print("\n✨ Key improvements:")
    print("• All archiving operations are now logged to the multiprocess log file")
    print("• Both console output AND log file capture the same information")
    print("• Progress tracking, errors, and success messages are all logged")
    print("• Multiprocess archiving steps are individually logged")
    print("• File cleanup and error handling are logged")
    
    # Clean up
    try:
        os.remove(log_filename)
        print(f"\n🧹 Cleaned up demo log file: {log_filename}")
    except:
        pass

if __name__ == "__main__":
    demo_logging()
