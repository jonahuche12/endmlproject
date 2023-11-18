import logging
import os
from datetime import datetime

# Create a 'logs' directory if it doesn't exist
logs_directory = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_directory, exist_ok=True)

# Define the log file path
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(logs_directory, log_file)

# Configure the logger
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s %(levelname)s %(message)s",  # Fixed the format
    level=logging.INFO,
)

# if __name__ == "__main__":
#     logging.info("Logging has Started")
