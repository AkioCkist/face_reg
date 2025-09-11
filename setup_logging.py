import os
import logging
from datetime import datetime

def setup_logging(log_name="face_recognition", level=logging.INFO):
    """Setup logging configuration with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d")
    log_filename = f"{log_name}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filepath

if __name__ == "__main__":
    # Test the logging setup
    logger, log_file = setup_logging("test")
    logger.info("Logging setup test successful")
    print(f"Log file created: {log_file}")