import paramiko
import os
import stat  # Add this import
from pathlib import Path
import getpass  # For secure password input
from dotenv import load_dotenv  # For loading environment variables
import re
from datetime import datetime, timezone
import glob

# Load environment variables from .env file
load_dotenv()

def download_folder_sftp():
    # Get configuration from environment variables or use defaults
    hostname = os.getenv('SFTP_HOSTNAME', "200.17.113.204")
    username = os.getenv('SFTP_USERNAME', "jpsantos")
    remote_path = os.getenv('SFTP_REMOTE_PATH', "Documents/sqw/logs/")
    local_path = os.getenv('SFTP_LOCAL_PATH', "./logs")
    
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Try to get password from environment variable first
        password = os.getenv('SFTP_PASSWORD')
        if not password:
            # Fallback to secure password input if not in environment
            password = getpass.getpass(f"Enter password for {username}@{hostname}: ")
        
        ssh.connect(hostname, username=username, password=password)
        
        # Get server timezone information
        server_timezone_info = get_server_timezone(ssh)
        
        sftp = ssh.open_sftp()
        
        def download_recursive(remote_dir, local_dir):
            try:
                files = sftp.listdir_attr(remote_dir)
                
                for file_attr in files:
                    remote_file = f"{remote_dir}/{file_attr.filename}"
                    local_file = os.path.join(local_dir, file_attr.filename)
                    
                    # Fix: Use stat.S_ISDIR instead of paramiko.sftp_attr.S_ISDIR
                    if stat.S_ISDIR(file_attr.st_mode):
                        Path(local_file).mkdir(parents=True, exist_ok=True)
                        print(f"Created directory: {local_file}")
                        download_recursive(remote_file, local_file)
                    else:
                        sftp.get(remote_file, local_file)
                        print(f"Downloaded: {remote_file} -> {local_file}")
                        
            except FileNotFoundError:
                print(f"Remote directory not found: {remote_dir}")
            except Exception as e:
                print(f"Error downloading {remote_dir}: {str(e)}")
        
        print(f"Starting download from {remote_path} to {local_path}")
        download_recursive(remote_path, local_path)
        
        sftp.close()
        ssh.close()
        print("Download completed successfully!")
        
        # Return server timezone info for analysis
        return server_timezone_info
        
    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your credentials.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def get_server_timezone(ssh):
    """
    Get server timezone information via SSH
    """
    try:
        print("\nDetecting server timezone...")
        
        # Try multiple methods to get timezone info
        timezone_commands = [
            "timedatectl show --property=Timezone --value",  # systemd systems
            "cat /etc/timezone",  # Debian/Ubuntu
            r"readlink /etc/localtime | sed 's/.*zoneinfo\///'",  # Most Linux systems
            "date +%Z",  # Timezone abbreviation
            "date +%z"   # UTC offset
        ]
        
        server_tz_info = {}
        
        for cmd in timezone_commands:
            try:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                output = stdout.read().decode().strip()
                if output and not stderr.read():
                    if "timedatectl" in cmd:
                        server_tz_info['timezone'] = output
                    elif "/etc/timezone" in cmd:
                        server_tz_info['timezone'] = output
                    elif "readlink" in cmd:
                        server_tz_info['timezone'] = output
                    elif "+%Z" in cmd:
                        server_tz_info['tz_abbrev'] = output
                    elif "+%z" in cmd:
                        server_tz_info['utc_offset'] = output
                    break
            except:
                continue
        
        # Get current server time
        try:
            stdin, stdout, stderr = ssh.exec_command("date '+%Y-%m-%d %H:%M:%S %Z %z'")
            server_time_output = stdout.read().decode().strip()
            if server_time_output:
                server_tz_info['current_time'] = server_time_output
        except:
            pass
        
        if server_tz_info:
            print(f"Server timezone detected: {server_tz_info}")
            return server_tz_info
        else:
            print("Could not detect server timezone automatically")
            return None
            
    except Exception as e:
        print(f"Error detecting server timezone: {str(e)}")
        return None

def analyze_heartbeat_logs(server_timezone_info=None):
    """
    Analyze heartbeat logs to determine expected vs actual heartbeats
    """
    local_path = os.getenv('SFTP_LOCAL_PATH', "./logs")
    
    # Find all log files
    log_files = glob.glob(os.path.join(local_path, "**/*.log"), recursive=True)
    
    if not log_files:
        print("No log files found to analyze")
        return
    
    print(f"Found {len(log_files)} log file(s)")
    
    # Ask user what they want to analyze
    print("\nWhat would you like to analyze?")
    print("1. All log files")
    print("2. Specific log file")
    print("3. Skip analysis")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print(f"\nAnalyzing all {len(log_files)} log files...")
            for log_file in log_files:
                print(f"\n=== Analyzing {os.path.basename(log_file)} ===")
                analyze_single_log(log_file, server_timezone_info)
            break
            
        elif choice == "2":
            # Show available log files
            print("\nAvailable log files:")
            # Group by directory for better organization
            log_dirs = {}
            for log_file in log_files:
                dir_name = os.path.dirname(log_file).replace(local_path, "").strip(os.sep)
                if not dir_name:
                    dir_name = "root"
                if dir_name not in log_dirs:
                    log_dirs[dir_name] = []
                log_dirs[dir_name].append(log_file)
            
            file_index = 1
            index_to_file = {}
            
            for dir_name in sorted(log_dirs.keys()):
                print(f"\n  {dir_name}/")
                for log_file in sorted(log_dirs[dir_name]):
                    filename = os.path.basename(log_file)
                    print(f"    {file_index:3d}. {filename}")
                    index_to_file[file_index] = log_file
                    file_index += 1
            
            # Get user selection
            while True:
                try:
                    file_choice = input(f"\nEnter file number (1-{len(log_files)}): ").strip()
                    file_num = int(file_choice)
                    if 1 <= file_num <= len(log_files):
                        selected_file = index_to_file[file_num]
                        print(f"\n=== Analyzing {os.path.basename(selected_file)} ===")
                        analyze_single_log(selected_file, server_timezone_info)
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(log_files)}")
                except ValueError:
                    print("Please enter a valid number")
            break
            
        elif choice == "3":
            print("Skipping log analysis.")
            break
            
        else:
            print("Please enter 1, 2, or 3")

def analyze_single_log(log_file_path, server_timezone_info=None):
    """
    Analyze a single log file for heartbeat information
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract heartbeat interval
        interval_match = re.search(r'Heartbeat interval: ([\d.]+)s', content)
        if not interval_match:
            print("Could not find heartbeat interval in log")
            return
        
        heartbeat_interval = float(interval_match.group(1))
        print(f"Heartbeat interval: {heartbeat_interval}s")
        
        # Find all heartbeats
        heartbeat_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - HEARTBEAT #(\d+) - Elapsed: ([\d.]+)s'
        heartbeats = re.findall(heartbeat_pattern, content)
        
        if not heartbeats:
            print("No heartbeats found in log")
            return
        
        print(f"Found {len(heartbeats)} heartbeats in log")
        
        # Parse timestamps and find the range
        first_heartbeat = heartbeats[0]
        last_heartbeat = heartbeats[-1]
        
        # Parse timestamps (assuming server timezone, you may need to adjust)
        first_time = datetime.strptime(first_heartbeat[0], "%Y-%m-%d %H:%M:%S,%f")
        last_time = datetime.strptime(last_heartbeat[0], "%Y-%m-%d %H:%M:%S,%f")
        
        print(f"First heartbeat: {first_time} (#{first_heartbeat[1]})")
        print(f"Last heartbeat:  {last_time} (#{last_heartbeat[1]})")
        
        # Calculate time difference
        time_diff = last_time - first_time
        total_seconds = time_diff.total_seconds()
        
        print(f"Time span: {total_seconds:.1f} seconds ({total_seconds/3600:.2f} hours)")
        
        # Calculate expected heartbeats
        # First heartbeat is usually immediate, then interval-based
        expected_heartbeats = 1 + int(total_seconds / heartbeat_interval)
        actual_heartbeats = len(heartbeats)
        
        print(f"Expected heartbeats: {expected_heartbeats}")
        print(f"Actual heartbeats:   {actual_heartbeats}")
        
        if actual_heartbeats < expected_heartbeats:
            missing = expected_heartbeats - actual_heartbeats
            print(f"⚠️  Missing {missing} heartbeat(s)")
        elif actual_heartbeats > expected_heartbeats:
            extra = actual_heartbeats - expected_heartbeats
            print(f"ℹ️  {extra} extra heartbeat(s) (normal variation)")
        else:
            print("✅ Heartbeat count matches expectation")
        
        # Check for gaps in heartbeat sequence
        heartbeat_numbers = [int(hb[1]) for hb in heartbeats]
        expected_sequence = list(range(1, len(heartbeats) + 1))
        
        if heartbeat_numbers != expected_sequence:
            print("⚠️  Heartbeat sequence has gaps or duplicates")
            missing_numbers = set(expected_sequence) - set(heartbeat_numbers)
            if missing_numbers:
                print(f"   Missing heartbeat numbers: {sorted(missing_numbers)}")
        else:
            print("✅ Heartbeat sequence is continuous")
        
        # Calculate current status based on download time
        download_time = datetime.now()
        
        # Automatic timezone adjustment using server info
        server_offset_hours = calculate_timezone_offset(server_timezone_info, download_time, last_time)
        
        if server_offset_hours is not None:
            print(f"\nAutomatic timezone adjustment:")
            print(f"Server timezone info: {server_timezone_info}")
            print(f"Calculated offset: {server_offset_hours:+.1f} hours")
        else:
            # Fallback to manual input if automatic detection failed
            print(f"\nTimezone adjustment (automatic detection failed):")
            print(f"Your local time: {download_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Last heartbeat:  {last_time.strftime('%Y-%m-%d %H:%M:%S')} (server time)")
            
            while True:
                try:
                    timezone_input = input("\nEnter timezone offset from server to your local time (hours, e.g., -3, 0, +2): ").strip()
                    if timezone_input == "":
                        server_offset_hours = 0
                        print("Using no timezone offset (assuming same timezone)")
                        break
                    else:
                        server_offset_hours = float(timezone_input)
                        if server_offset_hours > 0:
                            print(f"Server is {server_offset_hours} hours ahead of your local time")
                        elif server_offset_hours < 0:
                            print(f"Server is {abs(server_offset_hours)} hours behind your local time")
                        else:
                            print("Server and local time are in the same timezone")
                        break
                except ValueError:
                    print("Please enter a valid number (e.g., -3, 0, +2) or press Enter for no offset")
        
        # Calculate adjusted server time
        from datetime import timedelta
        adjusted_last_time = last_time + timedelta(hours=server_offset_hours)
        time_since_last = download_time - adjusted_last_time
        seconds_since_last = time_since_last.total_seconds()
        
        print(f"\nAdjusted analysis:")
        print(f"Last heartbeat (adjusted to local timezone): {adjusted_last_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current local time: {download_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time since last heartbeat: {seconds_since_last:.1f} seconds ({seconds_since_last/60:.1f} minutes)")
        
        if seconds_since_last > heartbeat_interval * 2:
            print("⚠️  Process may have stopped (no heartbeat for >2 intervals)")
        elif seconds_since_last > heartbeat_interval * 1.5:
            print("⚠️  Process may be delayed (no heartbeat for >1.5 intervals)")
        else:
            print("✅ Process appears to be running normally")
            
    except Exception as e:
        print(f"Error analyzing log file: {str(e)}")

def calculate_timezone_offset(server_tz_info, local_time, server_log_time):
    """
    Calculate timezone offset between server and local machine
    """
    if not server_tz_info:
        return None
    
    try:
        # Method 1: Parse UTC offset directly (e.g., "+0100", "-0500")
        if 'utc_offset' in server_tz_info:
            utc_offset_str = server_tz_info['utc_offset']
            if len(utc_offset_str) == 5 and (utc_offset_str[0] in ['+', '-']):
                sign = 1 if utc_offset_str[0] == '+' else -1
                hours = int(utc_offset_str[1:3])
                minutes = int(utc_offset_str[3:5])
                server_utc_offset = sign * (hours + minutes / 60.0)
                
                # Get local UTC offset
                import time
                local_utc_offset = -time.timezone / 3600.0
                if time.daylight and time.localtime().tm_isdst:
                    local_utc_offset += 1
                
                # Calculate the difference: how many hours to ADD to server time to get local time
                offset = local_utc_offset - server_utc_offset
                print(f"Server UTC offset: {server_utc_offset:+.1f}h, Local UTC offset: {local_utc_offset:+.1f}h")
                return offset
        
        # Method 2: Check if server is UTC and calculate based on that
        if 'timezone' in server_tz_info:
            server_tz = server_tz_info['timezone']
            if server_tz in ['UTC', 'Etc/UTC', 'GMT']:
                # Server is UTC, calculate local offset from UTC
                import time
                local_utc_offset = -time.timezone / 3600.0
                if time.daylight and time.localtime().tm_isdst:
                    local_utc_offset += 1
                
                # Server is at UTC (0), local offset tells us how many hours to add to UTC to get local time
                offset = local_utc_offset - 0  # same as just local_utc_offset
                print(f"Server is UTC (0h), Local UTC offset: {local_utc_offset:+.1f}h")
                return offset
        
        # Method 3: Parse current server time if available
        if 'current_time' in server_tz_info:
            # This would require more complex parsing of the server time format
            # For now, return None to fall back to manual input
            pass
        
        # If we can't calculate automatically, return None
        return None
        
    except Exception as e:
        print(f"Error calculating timezone offset: {str(e)}")
        return None

if __name__ == "__main__":
    server_timezone_info = download_folder_sftp()
    print("\n" + "="*60)
    print("HEARTBEAT ANALYSIS")
    print("="*60)
    analyze_heartbeat_logs(server_timezone_info)