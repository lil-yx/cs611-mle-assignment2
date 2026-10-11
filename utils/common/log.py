import logging
import os

from utils.common.config import PROJECT_ROOT

_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_FORMAT,
    datefmt=_DATEFMT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_LOG_DIR, "pipeline.log"), encoding="utf-8"),
    ],
)
logging.getLogger("py4j").setLevel(logging.WARNING)


def get_logger(name):
    return logging.getLogger(name)
