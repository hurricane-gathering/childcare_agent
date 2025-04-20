from typing import Optional, Dict, Any
from .api.llm_client import LLMClient
from .intent.classifier import IntentClassifier
from .memory.memory_manager import MemoryManager
from .utils.logger import logger
import sys


class ChildcareAssistant:
    def __init__(self):
        self.llm_client = LLMClient()
        self.intent_classifier = IntentClassifier()
        self.memory_manager = MemoryManager()
        logger.info("育儿小助手初始化完成")

    def _get_response_template(self, intent: str) -> str:
        """根据意图获取回复模板"""
        templates = {
            "询问年龄": "关于孩子的年龄，我建议：\n",
            "询问饮食": "关于孩子的饮食，以下是一些建议：\n",
            "询问睡眠": "关于孩子的睡眠，您可以考虑：\n",
            "询问教育": "关于孩子的教育问题，我的建议是：\n",
            "询问健康": "关于孩子的健康，需要注意：\n",
            "询问行为": "关于孩子的行为表现，建议：\n"
        }
        return templates.get(intent, "")

    def _prepare_prompt(self, user_input: str, intent: Optional[str], context: list) -> str:
        """准备发送给LLM的提示词"""
        template = self._get_response_template(intent) if intent else ""

        prompt = f"""你是一个专业的育儿顾问，请根据用户的问题提供专业、具体的建议。
当前问题：{user_input}

{template}请提供详细的建议，包括：
1. 具体可行的操作步骤
2. 注意事项
3. 可能遇到的问题及解决方案

请用温和、专业的语气回答，答案要具体实用。"""

        return prompt

    def process_message(self, user_input: str, stream: bool = True) -> str:
        """
        处理用户输入并生成回复

        Args:
            user_input: 用户输入的文本
            stream: 是否使用流式输出

        Returns:
            str: 助手的完整回复
        """
        try:
            # 1. 意图识别
            intent_result = self.intent_classifier.classify(user_input)
            intent = intent_result[0] if intent_result else None
            logger.info(f"识别到意图: {intent}")

            # 2. 获取上下文
            context = self.memory_manager.get_context()

            # 3. 准备消息
            messages = [{"role": "user", "content": self._prepare_prompt(
                user_input, intent, context)}]

            # 4. 调用LLM获取回复
            if stream:
                # 流式输出
                full_response = ""
                print("\n助手: ", end="", flush=True)
                for chunk in self.llm_client.chat(messages, stream=True):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # 换行
            else:
                # 普通输出
                full_response = self.llm_client.chat(messages, stream=False)

            # 5. 保存对话记录
            self.memory_manager.add_memory("user", user_input, intent)
            self.memory_manager.add_memory("assistant", full_response)

            return full_response

        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")
            return "抱歉，我现在遇到了一些问题。请稍后再试。"


def main():
    """主函数"""
    assistant = ChildcareAssistant()
    logger.info("育儿小助手启动成功")

    print("育儿小助手已启动，请输入您的问题（输入'退出'结束对话）")

    while True:
        try:
            user_input = input("\n您的问题: ").strip()

            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("感谢使用育儿小助手，再见！")
                break

            if not user_input:
                continue

            assistant.process_message(user_input, stream=True)

        except KeyboardInterrupt:
            print("\n程序被中断，正在退出...")
            break
        except Exception as e:
            logger.error(f"运行时错误: {str(e)}")
            print("发生错误，请重试。")


if __name__ == "__main__":
    main()
