import sys
from src.logger import logging

# This function is used to create a custom error message that includes the file name, line number, and error message
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(file_name,exc_tb.tb_lineno,str(error))

    return error_message

    
# This class is used to create a custom exception that can be used to handle errors in the project
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    

# test custom exception
# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         # put the error message in the log file
#         logging.info("Divide by zero error occurred.")
#         raise CustomException(e,sys)