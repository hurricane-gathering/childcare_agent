from loguru import logger
import sys
import os
from datetime import datetime


def setup_logger():
    # 确保日志目录存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名（按日期）
    log_file = os.path.join(
        log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    # 移除默认的处理器
    logger.remove()

    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # 添加文件处理器
    logger.add(
        log_file,
        rotation="00:00",  # 每天轮转
        retention="30 days",  # 保留30天
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        encoding="utf-8"
    )

    return logger


# 初始化日志配置
logger = setup_logger()
