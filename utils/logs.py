import logging

def initLogging(file_name, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create handlers
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(file_name)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('[%(asctime)s] %(levelname)s | %(message)s')
    f_format = logging.Formatter('[%(asctime)s] %(levelname)s | %(message)s')
    s_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)