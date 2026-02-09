import logging
import os

def configure_logger(level: int = logging.INFO, name: str = "agentflow") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()  # clear existing handlers

    # log to stdout
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging
    return logger


def condprint(*args, **kwargs):
    print_flag = os.environ.get('PRINT_FLAG', '').lower()
    
    if print_flag == 'true':
        print(*args, **kwargs)
