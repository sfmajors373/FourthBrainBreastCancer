import logging
from logging.handlers import TimedRotatingFileHandler
import sys

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

MAP_LOGGING = {1: logging.DEBUG, 2: logging.WARNING}


def get_logging_level(level=1):
    LOGGING_LEVEL = MAP_LOGGING[level]
    return LOGGING_LEVEL


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(file_name):
    file_handler = TimedRotatingFileHandler(file_name, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logging_path, logging_level=logging.WARNING):
    logger_name = logging_path.split('/')[-1].split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(logging_path))
    logger.propagate = False
    return logger
