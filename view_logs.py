import os
import glob
from datetime import datetime

def view_logs():
    """View recent log files"""
    logs_dir = "logs"
    
    if not os.path.exists(logs_dir):
        print("No logs directory found. Run the face recognition system first.")
        return
    
    # Find all log files
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    
    if not log_files:
        print("No log files found.")
        return
    
    # Sort by modification time (newest first)
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    print("Available log files:")
    print("-" * 50)
    
    for i, log_file in enumerate(log_files):
        file_size = os.path.getsize(log_file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        filename = os.path.basename(log_file)
        
        print(f"{i+1}. {filename}")
        print(f"   Size: {file_size} bytes")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    try:
        choice = input("Enter log file number to view (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(log_files):
            selected_file = log_files[choice_idx]
            print(f"\nViewing: {os.path.basename(selected_file)}")
            print("=" * 80)
            
            with open(selected_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        else:
            print("Invalid selection.")
            
    except (ValueError, IndexError):
        print("Invalid input.")
    except Exception as e:
        print(f"Error reading log file: {e}")

if __name__ == "__main__":
    view_logs()