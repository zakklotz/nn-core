import logging

def get_logger(name: str = "nncore", level: str | int = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    # Convert level string to numeric if needed
    if isinstance(level, str):
        level_name = level.upper()
        level = getattr(logging, level_name, None)
        if not isinstance(level, int):
            raise ValueError(f"Invalid log level: {level_name!r}")

    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent double logging from root logger
    logger.propagate = False

    return logger