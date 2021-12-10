import logging

logging.basicConfig(
    level=logging.NOTSET
)


def get_logger(name):
    return logging.getLogger(name=name)
