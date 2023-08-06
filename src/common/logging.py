import logging
from pathlib import Path
from common.desing_patterns import Singleton



class Logger(metaclass=Singleton):
    """
    Logger is a singleton class.
    """
    
    def __init__(self, filename="log", dir=".", name="main_logger"):
        filepath = f"{dir}/{filename}.log"
        Path(dir).mkdir(exist_ok=True, parents=True)

        logger = logging.getLogger(name=name)

        file_handler = logging.FileHandler(
            filename=filepath,
            encoding="utf-8",
            mode="a")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s) %(filename)s:%(lineno)d => %(message)s"
        ))
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO) 

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        self.__logger = logger     
        return
    
    def get_logger(self):
        return self.__logger
    
    
def get_logger(filename="log", dir="."):
    # filepath = f"{dir}/{filename}.log"
    obj = Logger(filename=filename, dir=dir, name="main_logger")
    return obj.get_logger()
