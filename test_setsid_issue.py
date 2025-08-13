#!/usr/bin/env python3

"""
Test script to diagnose the os.setsid issue
"""

import os
import sys
import subprocess

def test_setsid():
    """Test if os.setsid works on this system"""
    print("Testing os.setsid functionality...")
    
    try:
        # Test if os.setsid exists and can be called
        if hasattr(os, 'setsid'):
            print("✓ os.setsid is available")
            
            # Test creating a subprocess with os.setsid
            test_script = '''
import os
import time
print(f"Test process PID: {os.getpid()}")
print(f"Test process SID: {os.getsid(0)}")
time.sleep(2)
print("Test process completed")
'''
            
            with open('test_setsid_subprocess.py', 'w') as f:
                f.write(test_script)
            
            try:
                process = subprocess.Popen(
                    [sys.executable, 'test_setsid_subprocess.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                
                stdout, stderr = process.communicate(timeout=5)
                
                if process.returncode == 0:
                    print("✓ os.setsid works in subprocess")
                    print("Output:", stdout.decode())
                else:
                    print("✗ subprocess with os.setsid failed")
                    print("Error:", stderr.decode())
                    
            except Exception as e:
                print(f"✗ Error testing subprocess with os.setsid: {e}")
                return False
                
            finally:
                # Clean up test file
                try:
                    os.remove('test_setsid_subprocess.py')
                except:
                    pass
                    
        else:
            print("✗ os.setsid is not available on this system")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing os.setsid: {e}")
        return False

def test_alternative_background():
    """Test alternative background execution methods"""
    print("\\nTesting alternative background methods...")
    
    # Test simple subprocess without setsid
    try:
        with open('test_background_simple.log', 'w') as log_f:
            process = subprocess.Popen(
                [sys.executable, '-c', 'import time; print("Simple test"); time.sleep(1); print("Done")'],
                stdout=log_f,
                stderr=subprocess.STDOUT
            )
        
        process.wait(timeout=3)
        
        if process.returncode == 0:
            print("✓ Simple subprocess works")
            with open('test_background_simple.log', 'r') as f:
                print("Output:", f.read().strip())
        else:
            print("✗ Simple subprocess failed")
            
    except Exception as e:
        print(f"✗ Error with simple subprocess: {e}")
    
    finally:
        try:
            os.remove('test_background_simple.log')
        except:
            pass
    
    # Test with nohup
    try:
        with open('test_nohup.log', 'w') as log_f:
            process = subprocess.Popen(
                ['nohup', sys.executable, '-c', 'import time; print("Nohup test"); time.sleep(1); print("Done")'],
                stdout=log_f,
                stderr=subprocess.STDOUT
            )
        
        process.wait(timeout=3)
        
        if process.returncode == 0:
            print("✓ nohup works")
        else:
            print("✗ nohup failed or not available")
            
    except Exception as e:
        print(f"✗ Error with nohup: {e}")
    
    finally:
        try:
            os.remove('test_nohup.log')
        except:
            pass

if __name__ == "__main__":
    print("=== Background Execution Diagnostics ===")
    print(f"Platform: {os.name}")
    print(f"Python: {sys.executable}")
    print(f"PID: {os.getpid()}")
    
    setsid_works = test_setsid()
    test_alternative_background()
    
    print("\\n=== Recommendations ===")
    if setsid_works:
        print("✓ os.setsid should work - the issue might be elsewhere")
    else:
        print("✗ os.setsid has issues - use alternative background methods")
        print("  Recommended: Use safe_background_launcher.py")
        print("  Alternative: Use screen/tmux or nohup directly")
