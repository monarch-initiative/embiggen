import os
import sys
import inspect
import logging
from colorama import init, Fore, Style
import linecache

####################################################################################################
# Configs
####################################################################################################

levels = {
    "debug": (logging.DEBUG, Fore.CYAN),
    "info": (logging.INFO, Fore.RESET),
    "warning": (logging.WARNING, Fore.YELLOW),
    "error": (logging.ERROR, Fore.MAGENTA),
    "critical": (logging.CRITICAL, Fore.RED),
}

stream = sys.stdout
stream_format = "%(asctime)s %(color)s[%(levelname)-8s] %(script)+10s-%(function)+8s-%(line_number)03d: %(msg)s%(reset_color)s"
files_format = "%(asctime)s [%(levelname)-8s] %(script)+10s-%(function)+8s-%(line_number)03d: %(msg)s"

####################################################################################################
# Logger definition
####################################################################################################


class MyLogger(logging.Logger):
    """Custom template class which will be filled dynamically by the 
        register_method function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def define_method(level_name, level, color):
    """Function that build the custom loggin method from the parameters supplied
        by the user."""
    _level_name = level_name
    _level, _color = level, color

    def wrapped(self, msg, args=tuple(), **kwargs):
        kwargs.setdefault("extra", dict())
        kwargs["extra"]["color"] = _color
        kwargs["extra"]["reset_color"] = Style.RESET_ALL
        caller_stack = inspect.stack()[2]
        kwargs["extra"]["function"] = caller_stack.function
        kwargs["extra"]["line_number"] = caller_stack.lineno
        kwargs["extra"]["script"] = os.path.basename(caller_stack.filename)
        if self.isEnabledFor(_level):
            self._log(_level, msg, args, **kwargs)
    return wrapped


def register_method(logger_class, level_name, level, color):
    method = define_method(level_name, level, color)
    setattr(logger_class, level_name, method)


def formatException():
    """Format exceptions nicely"""
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    return f' - line : {lineno} [ {line.strip()} ] - {exc_obj}'


def register_exit_method(logger_class, level_name, level, color):
    """Method to add custom logging methods that does sys.exit after logging"""
    method = define_method(level_name, level, color)

    def wrapped(self, error_code, *args, **kwargs):
        method(self, *args, **kwargs)
        sys.exit(int(error_code))
    setattr(logger_class, level_name + "_exit", wrapped)


def register_methods(logger_class, levels):
    """Method to add custom logging methods"""
    for level_name, (level, color) in levels.items():
        register_method(logger_class, level_name, level, color)
        register_exit_method(logger_class, level_name, level, color)


register_methods(MyLogger, levels)

####################################################################################################
# Select the level to log per file
####################################################################################################


class SelectiveFilter(logging.Filter):
    def __init__(self, levels):
        self.__levels = levels

    def filter(self, logRecord):
        return logRecord.levelno in self.__levels

####################################################################################################
# Logger Initializzation
####################################################################################################


logger = MyLogger("root_logger")
stream_handler = logging.StreamHandler(stream)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(stream_format))
logger.addHandler(stream_handler)
