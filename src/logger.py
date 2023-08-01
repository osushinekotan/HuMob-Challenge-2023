import logging
import time
from contextlib import contextmanager

from colorlog import ColoredFormatter


class Logger:
    def __init__(self, name: str = "", filename=None, level=logging.INFO, filemode="a"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Define formatter
        log_format = "[%(asctime)s] %(log_color) s[%(name)s] [%(levelname)s] - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = ColoredFormatter(
            log_format,
            datefmt=date_format,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Check if the logger already has handlers. If not, add new handlers.
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            if filename is not None:
                file_handler = logging.FileHandler(filename, mode=filemode)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    @contextmanager
    def time_log(self, target: str | None = None) -> None:
        target = self.logger.name if target is None else target
        start_time = time.time()
        self.info(f"start {target} 🚀")
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.info(f"end {target} ✨ - elapsed time: {elapsed_time:.2f} seconds ⏰")


# examples
if __name__ == "__main__":
    logger = Logger()

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    with logger.time_log("Some Task"):
        time.sleep(2)
