import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from ..config.settings import get_settings
from ..utils.logger import logger


class MemoryManager:
    def __init__(self):
        self.settings = get_settings()
        self.memory_file = self.settings.MEMORY_FILE
        self.max_items = self.settings.MAX_MEMORY_ITEMS
        self.short_term_memory: List[Dict] = []
        self.long_term_memory: List[Dict] = []
        self._load_memory()

    def _load_memory(self):
        """从文件加载长期记忆"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.long_term_memory = json.load(f)
                logger.info(f"成功加载{len(self.long_term_memory)}条长期记忆")
            else:
                # 确保目录存在
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                self.long_term_memory = []
                logger.info("创建新的记忆存储")
        except Exception as e:
            logger.error(f"加载记忆文件时出错: {str(e)}")
            self.long_term_memory = []

    def _save_memory(self):
        """保存长期记忆到文件"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f,
                          ensure_ascii=False, indent=2)
            logger.info("成功保存记忆到文件")
        except Exception as e:
            logger.error(f"保存记忆文件时出错: {str(e)}")

    def add_memory(self, role: str, content: str, intent: Optional[str] = None):
        """
        添加新的记忆

        Args:
            role: 说话者角色 ("user" 或 "assistant")
            content: 对话内容
            intent: 用户意图（可选）
        """
        memory_item = {
            "role": role,
            "content": content,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        }

        # 添加到短期记忆
        self.short_term_memory.append(memory_item)
        if len(self.short_term_memory) > 10:  # 保持最近10条对话
            self.short_term_memory.pop(0)

        # 添加到长期记忆
        self.long_term_memory.append(memory_item)
        if len(self.long_term_memory) > self.max_items:
            self.long_term_memory.pop(0)

        # 保存到文件
        self._save_memory()
        logger.info(f"添加新记忆: {role} - {content[:50]}...")

    def get_context(self, max_items: int = 5) -> List[Dict[str, str]]:
        """
        获取当前对话上下文

        Args:
            max_items: 返回的最大记忆条数

        Returns:
            List[Dict[str, str]]: 格式化的上下文记忆列表
        """
        # 优先使用短期记忆
        context = [
            {"role": item["role"], "content": item["content"]}
            for item in self.short_term_memory[-max_items:]
        ]
        return context

    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """
        搜索历史记忆

        Args:
            query: 搜索关键词
            limit: 返回结果数量限制

        Returns:
            List[Dict]: 匹配的记忆列表
        """
        matches = []
        for memory in reversed(self.long_term_memory):
            if query.lower() in memory["content"].lower():
                matches.append(memory)
                if len(matches) >= limit:
                    break
        return matches

    def get_memory_by_intent(self, intent: str, limit: int = 5) -> List[Dict]:
        """
        根据意图检索记忆

        Args:
            intent: 意图标识
            limit: 返回结果数量限制

        Returns:
            List[Dict]: 匹配意图的记忆列表
        """
        matches = []
        for memory in reversed(self.long_term_memory):
            if memory.get("intent") == intent:
                matches.append(memory)
                if len(matches) >= limit:
                    break
        return matches

    def clear_short_term_memory(self):
        """清空短期记忆"""
        self.short_term_memory = []
        logger.info("已清空短期记忆")
