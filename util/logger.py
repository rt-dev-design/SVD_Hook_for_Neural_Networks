import logging
import os
from datetime import datetime
import sys
import traceback

def build_logger(name, log_dir, log_filename=None, use_this_logger_for_global_exceptions=False, logging_level=logging.DEBUG):
    """创建一个 logger，同时输出到控制台和文件"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 默认文件名：log_2025-09-06_12-34-56.log
    if log_filename is None:
        log_filename = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    log_path = os.path.join(log_dir, log_filename)

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)  # 可改为 DEBUG/INFO/WARNING/ERROR

    # 避免重复添加 handler
    if not logger.handlers:
        # 文件 handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging_level)

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)

        # 日志格式
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    if use_this_logger_for_global_exceptions:
        def use_this_logger_for_global_exceptions(exc_type, exc_value, exc_traceback):
            """
            自定义全局异常处理，将未捕获异常写入日志
            """
            if issubclass(exc_type, KeyboardInterrupt):
                # 对 Ctrl+C 不做特殊处理，走默认逻辑
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.error(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        # 替换默认 excepthook
        sys.excepthook = use_this_logger_for_global_exceptions

    return logger
