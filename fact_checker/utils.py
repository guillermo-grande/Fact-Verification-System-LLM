import os
import logging
from typing import Literal, Union

def load_prompt(type: str, version: int) -> str:
    """Given the type of a prompt and its version, this function returns the prompt. 

    Args:
        type (str): prompt type. ejem. decompose, generate, claim
        version (int): prompt version

    Returns:
        str: text of the prompt in the file `{type}/{type}-v{version}.txt`
    """
    file_name = f"./prompts/{type}/{type}-v{version}.txt"
    if not os.path.exists(file_name):
        raise FileNotFoundError
    
    with open(file_name, 'r') as prompt_file:
        content = prompt_file.read()
    return content

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

LOGGER_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
def configure_logger(logger: logging.Logger, level: LOGGER_LEVELS = 'INFO') -> logging.Logger:
    """Configure the logger with the default streams and format

    Args:
        logger (logging.Logger): logger to configure
        level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", optional): level of logging. Defaults to 'INFO'.

    Returns:
        logging.Logger: configured logger
    """
    stream = logging.StreamHandler()
    stream.setFormatter(CustomFormatter())
    logger.addHandler(stream)
    logger.setLevel(level)
    return logger