from datetime import datetime
import logging

current_datetime = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")

logging.basicConfig(
    filename=current_datetime + ".log",
    format='%(asctime)s %(message)s',
    filemode='w'
)


def get_logger(name):
    return logging.getLogger(name=name)
