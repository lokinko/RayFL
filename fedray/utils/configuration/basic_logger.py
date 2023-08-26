import os
import yaml
from dotenv import load_dotenv
from logging.config import dictConfig


def load_configuration(log_path=None):
    """_summary_

    Args:
        log_path (_type_, optional): _description_. Defaults to None.
    """
    # Load environment variables
    load_dotenv()

    # Load logging configuration
    with open(os.environ["LOGGING_FILE_PATH"], "r") as file:
        logger_config = yaml.safe_load(file)
        if log_path:
            logger_config["handlers"]["file"]["filename"] = log_path
        dictConfig(logger_config)


if __name__ == "__main__":
    load_configuration()
