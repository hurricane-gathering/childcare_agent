from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..config.settings import get_settings
from ..utils.logger import logger


class IntentClassifier:
    def __init__(self):
        self.settings = get_settings()
        self.threshold = self.settings.INTENT_THRESHOLD
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

        # 预定义意图和对应的关键词/模式
        self.intent_patterns = {
            "询问年龄": [
                r"几岁",
                r"多大",
                r"年龄",
                r"(?:今年|现在).*岁"
            ],
            "询问饮食": [
                r"吃",
                r"喝",
                r"饭",
                r"餐",
                r"食物",
                r"营养"
            ],
            "询问睡眠": [
                r"睡觉",
                r"睡眠",
                r"作息",
                r"休息"
            ],
            "询问教育": [
                r"学习",
                r"教育",
                r"培养",
                r"兴趣",
                r"才艺"
            ],
            "询问健康": [
                r"生病",
                r"发烧",
                r"感冒",
                r"疫苗",
                r"体检"
            ],
            "询问行为": [
                r"表现",
                r"行为",
                r"习惯",
                r"性格"
            ]
        }

        # 编译正则表达式
        self.compiled_patterns = {
            intent: [re.compile(pattern) for pattern in patterns]
            for intent, patterns in self.intent_patterns.items()
        }

        # 为每个意图准备示例句子
        self.intent_examples = {
            "询问年龄": [
                "孩子现在几岁了",
                "你能告诉我孩子多大了吗",
                "想知道孩子的年龄"
            ],
            "询问饮食": [
                "孩子应该吃什么",
                "如何安排孩子的饮食",
                "孩子的营养该怎么搭配"
            ],
            "询问睡眠": [
                "孩子晚上几点睡觉比较好",
                "如何培养良好的作息习惯",
                "孩子睡眠质量不好怎么办"
            ],
            "询问教育": [
                "孩子该上什么兴趣班",
                "如何培养孩子的学习兴趣",
                "孩子教育应该注意什么"
            ],
            "询问健康": [
                "孩子发烧了怎么办",
                "多久带孩子体检一次",
                "孩子感冒了该如何护理"
            ],
            "询问行为": [
                "孩子总是不听话怎么办",
                "如何纠正孩子的坏习惯",
                "孩子性格内向该怎么引导"
            ]
        }

        # 训练TF-IDF向量化器
        all_examples = []
        for examples in self.intent_examples.values():
            all_examples.extend(examples)
        self.vectorizer.fit(all_examples)

    def _rule_based_match(self, text: str) -> List[Tuple[str, float]]:
        """基于规则的意图匹配"""
        matches = []
        for intent, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text):
                    score += 1
            if score > 0:
                confidence = min(score / len(patterns), 1.0)
                matches.append((intent, confidence))
        return matches

    def _similarity_based_match(self, text: str) -> List[Tuple[str, float]]:
        """基于文本相似度的意图匹配"""
        text_vector = self.vectorizer.transform([text])
        matches = []

        for intent, examples in self.intent_examples.items():
            example_vectors = self.vectorizer.transform(examples)
            similarities = cosine_similarity(text_vector, example_vectors)
            max_similarity = np.max(similarities)
            if max_similarity > self.threshold:
                matches.append((intent, float(max_similarity)))

        return matches

    def classify(self, text: str) -> Optional[Tuple[str, float]]:
        """
        对输入文本进行意图分类

        Args:
            text: 输入文本

        Returns:
            Tuple[str, float]: 意图和置信度的元组，如果没有匹配则返回None
        """
        try:
            logger.info(f"开始对文本进行意图分类: {text}")

            # 获取规则匹配和相似度匹配的结果
            rule_matches = self._rule_based_match(text)
            similarity_matches = self._similarity_based_match(text)

            # 合并两种方法的结果
            all_matches = {}
            for intent, score in rule_matches + similarity_matches:
                if intent in all_matches:
                    all_matches[intent] = max(all_matches[intent], score)
                else:
                    all_matches[intent] = score

            if not all_matches:
                logger.info("没有找到匹配的意图")
                return None

            # 选择置信度最高的意图
            best_intent = max(all_matches.items(), key=lambda x: x[1])
            logger.info(f"识别到意图: {best_intent[0]}，置信度: {best_intent[1]:.2f}")

            return best_intent if best_intent[1] >= self.threshold else None

        except Exception as e:
            logger.error(f"意图分类过程中出错: {str(e)}")
            raise
