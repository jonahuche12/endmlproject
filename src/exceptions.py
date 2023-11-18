import sys
import traceback
from src.logger import logging

def error_message_details(error, error_details: sys):
    _, _, err_tb = sys.exc_info()
    filename = err_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script [{0}] line no [{1}] error message [{2}]".format(
        filename, err_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_details: sys):
        super().__init__(str(error))
        self.error_message = error_message_details(error, error_details=error_details)

    def __str__(self):
        return self.error_message

# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.error("Divide by zero error")
#         raise CustomException(e, sys)
