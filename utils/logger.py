# -*- coding: UTF-8 -*-
# logger module

import logging

LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"

class Logger():
    def __init__(self, file=None):
        if file is not None:
            logging.basicConfig(filename=file, level=logging.DEBUG, format=LOG_FORMAT)
        else:
            logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

        self.loggers = {
            "INFO" : logging.info,
            "DEBUG" : logging.debug,
            "WARNING" : logging.warning,
            "ERROR" : logging.error,
            "CRITICAL" : logging.critical,
        }

    def log(self, level:str, message:str):
        assert level in self.loggers, "Logging level error!"
        self.loggers[level](message)