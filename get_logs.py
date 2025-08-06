import paramiko
import os
import stat  # Add this import
from pathlib import Path
import getpass  # For secure password input
from dotenv import load_dotenv  # For loading environment variables

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
        
    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your credentials.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    download_folder_sftp()