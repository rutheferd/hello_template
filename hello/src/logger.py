import logging

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename = "logs.log",
    level = logging.INFO
)
global logger

logger = logging.getLogger()

print("DONE")