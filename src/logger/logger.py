import logging

from pathlib import Path

class Logger:
    def __init__(self, logger_name: str, console_stream: bool) -> None:
        """
        Logger class

        :param logger_name: Name of the logger to be displayed in logs
        :param console_stream: True for logger to log not only to respective file, but also to console
        """
        self.logger_name = logger_name
        self.console_stream = console_stream

        self.__configure()

    def __configure(self) -> None:
        self.logger = logging.getLogger(self.logger_name)

        if self.logger.handlers:
            return

        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Ensure logs directory exists one level outside src/
        log_dir = Path(__file__).resolve().parents[2] / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'{self.logger_name}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if self.console_stream:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

        self.log(f'Logger {self.logger_name} initialized', level="info")

    def log(self, message: str, level: str = None) -> None:
        if level is None:
            level = "warning"
            message = f'[LOG WITHOUT SPECIFIED LEVEL] - {message}'

        levels = {
            "info": self.logger.info,
            "warning": self.logger.warning,
            "error": self.logger.error,
            "debug": self.logger.debug,
            "critical": self.logger.critical,
        }

        method = levels[level]

        method(message)

    def shutdown(self) -> None:
        """
        Removes all handlers from the logger and disables further logging.
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()

    def __str__(self) -> str:
        """
        String representation of the logger
        """
        return f"Logger(name={self.logger.name}, level={self.logger.level})"
    