import logging
import os

def configure_logger(log_filename='app.logs', log_level=logging.DEBUG):
    """
    Konfiguruje logger zapisujący logi tylko do pliku .logs.

    Args:
        log_filename (str): Nazwa pliku logów (domyślnie 'app.logs')
        log_level: Poziom logowania (domyślnie logging.DEBUG)

    Returns:
        logging.Logger: Skonfigurowany logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Usuwamy istniejące handlery, jeśli są (by nie dublować logów)
    logger.handlers.clear()

    # Format logów z nazwą pliku i linią kodu
    format_str = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    file_formatter = logging.Formatter(format_str, datefmt=date_format)

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    return logger
