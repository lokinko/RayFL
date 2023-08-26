import os
import logging
from fedray.utils.configuration import load_configuration

if __name__ == "__main__":
    load_configuration()
    logger = logging.getLogger("root")
    logger.info("Test")
