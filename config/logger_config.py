# logger_config.py
import logging
import sys

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all logs

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler("logs/app.log", mode="a")
    fh.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Avoid adding duplicate handlers in interactive sessions
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
