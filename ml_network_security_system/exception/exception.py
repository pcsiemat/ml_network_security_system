# ''' import sys
# from ml_network_security_system.logger.log import logging
# class NetworkSecurityException(Exception):
#     def __init__(self,error_message,error_details:sys):
#         self.error_message = error_message
#         _,_,exc_tb = error_details.exc_info()
        
#         self.lineno=exc_tb.tb_lineno
#         self.file_name=exc_tb.tb_frame.f_code.co_filename 
    
#     def __str__(self):
#         return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
#         self.file_name, self.lineno, str(self.error_message))
        
# if __name__=='__main__':
#     try:
#         logging.info("Enter the try block")
#         a=1/0
#         print("This will not be printed",a)
#     except Exception as e:
#            raise NetworkSecurityException(e,sys) '''

import os
import sys
from ml_network_security_system.logger.log import logging

class AppException(Exception):
    """
    AppException is customized exception class designed to capture refined details about exception
    such as python script file line number along with error message
    With custom exception one can easily spot source of error and provide quick fix.
    
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = AppException.error_message_detail(error_message, error_detail=error_detail)

        logging.error("=" * 80)
        logging.error(f"❌ EXCEPTION: {type(error_message).__name__}")
        logging.error(self.error_message)
        logging.error("=" * 80)



    @staticmethod
    def error_message_detail(error:Exception, error_detail:sys):
        """
        error: Exception object raise from module
        error_detail: is sys module contains detail information about system execution information.
        """
        _, _, exc_tb = error_detail.exc_info()
        #extracting file name from exception traceback
        file_name = exc_tb.tb_frame.f_code.co_filename 

        #preparing error message
        error_message = f"Error occurred python script name [{file_name}]" \
                        f" line number [{exc_tb.tb_lineno}] error message [{error}]."

        return error_message

    def __repr__(self):
        """
        Formating object of AppException
        """
        return AppException.__name__.__str__()

    def __str__(self):
        """
        Formating how a object should be visible if used in print statement.
        """
        return self.error_message


if __name__=='__main__':
    try:
        a=1/0
        print("This will not be printed",a)
    except Exception as e:
        raise AppException(e, sys) from e


