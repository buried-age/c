# -*- coding: utf-8 -*-
import logging


def get_logger(name, file, level):
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)

    file_handler = logging.FileHandler(filename=file, encoding="utf-8")
    file_handler.setLevel(level=level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s : %(message)s - (%(lineno)s)")
    file_handler.setFormatter(fmt=formatter)
    stream_handler.setFormatter(fmt=formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
