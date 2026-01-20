import pickle, os
import logging, colorlog, sys
import logging.handlers as handlers
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import json

stdout = colorlog.StreamHandler(stream=sys.stdout)


class Logger(object):
    def __init__(
        self,
        logfile="main.log",  # set to None if you dont want to save the Logger into a logfile
        console=True,
        verbose=False,
    ):
        super().__init__()
        self.logfile = logfile
        if verbose:
            self.level = logging.DEBUG
        else:
            self.level = logging.INFO

        assert console == True or logfile is not None, (
            "At least enable one from console or logfile for Logger"
        )
        assert isinstance(logfile, str) or logfile is None, (
            "Provided logfile arg must be a string (filename) or None."
        )

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        self.enabled = True

        if console:
            self.set_logging_into_console()
        if logfile is not None:
            self.set_logging_into_file()

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def set_logging_into_console(self):
        """Set logging into console."""
        ch_formatter = CustomFormatter(
            "%(white)s%(asctime)s%(reset)s >>> %(log_color)s%(message)s%(reset)s"
        )  # %(name)s:   # | %(blue)s%(filename)s:%(lineno)s%(reset)s   # %(log_color)s%(levelname)s%(reset)s
        ch = colorlog.StreamHandler(stream=sys.stdout)  # logging.StreamHandler()
        ch.setLevel(self.level)
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)

    def set_logging_into_file(self):
        """Set logging into file."""
        fh_formatter = logging.Formatter("%(asctime)s | %(levelname)s >>> %(message)s")
        fh = handlers.RotatingFileHandler(
            self.logfile, maxBytes=(1024**2 * 2), backupCount=3
        )
        fh.setLevel(self.level)
        fh.setFormatter(fh_formatter)
        self.logger.addHandler(fh)

    def enable_logging_into_file(self, logfile=None):
        """Enable logging into file."""
        if logfile is not None:
            self.logfile = logfile

        if self.logfile is not None:
            self.set_logging_into_file()
        else:
            raise ValueError("Logfile is not set. Cannot enable logging into file.")

    def disable_logging_into_file(self):
        """Disable logging into file."""
        for handler in self.logger.handlers:
            if isinstance(handler, handlers.RotatingFileHandler):
                self.logger.removeHandler(handler)
                handler.close()
                break

    def debug(self, message):
        if not self.enabled:
            return
        self.logger.debug(message)

    def info(self, message):
        if not self.enabled:
            return
        self.logger.info(message)

    def warning(self, message):
        if not self.enabled:
            return
        self.logger.warning(message)

    def error(self, message):
        if not self.enabled:
            return
        self.logger.error(message)

    def critical(self, message):
        if not self.enabled:
            return
        self.logger.critical(message)


class CustomFormatter(colorlog.ColoredFormatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"

    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    format = "%(asctime)s - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: light_blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    # for level, color in zip(("info", "warn", "error", "debug"), (green, yellow, red, light_blue)):
    #     setattr(logger, level, add_color(getattr(logger, level), color))


#


class ResultsDictLogger:
    def __init__(self, filename="results", filepath="data", use_json=True):
        # self._find_root_folder()
        self.use_json = use_json

        if filepath is not None:
            if not os.path.exists(filepath):
                os.mkdir(filepath)
        else:
            filepath = os.getcwd()

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        if self.use_json:
            self.fname = os.path.join(filepath, f"{filename}.json")
        else:
            self.fname = os.path.join(filepath, f"{filename}.pkl")

    # def _find_root_folder(self):
    #     backward_lvl = 0
    #     got_root = False
    #     while got_root == False:
    #         self.root_folder = str(Path(os.getcwd()).parents[backward_lvl])
    #         if os.path.exists(os.path.join(self.root_folder,
    #                                        'results')):
    #             got_root = True
    #         else:
    #             backward_lvl = backward_lvl + 1
    #         if backward_lvl > 10:
    #             break

    def save(self, uid, key, result):
        """Save PyTorch model in path"""
        if self.use_json:
            if os.path.exists(self.fname):
                with open(self.fname, "r") as f:
                    results_log = json.load(f)
            else:
                results_log = OrderedDict()
        else:
            if os.path.exists(self.fname):
                with open(self.fname, "rb") as f:
                    results_log = pickle.load(f)
            else:
                results_log = OrderedDict()

        if uid not in results_log.keys():
            results_log[uid] = {}

        results_log[uid]["_timestamp"] = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        results_log[uid][key] = result

        if self.use_json:
            with open(self.fname, "w") as f:
                json.dump(results_log, f, indent=4, sort_keys=True)
        else:
            with open(self.fname, "wb") as f:
                pickle.dump(results_log, f, pickle.HIGHEST_PROTOCOL)
