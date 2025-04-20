from typing import List, Dict, Any, Generator, Optional
import openai
from openai import OpenAI
from ..config.settings import get_settings
from ..utils.logger import logger


class LLMClient:
    def __init__(self):
        self.settings = get_settings()

        # 配置OpenAI客户端
        self.client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_API_BASE,
            timeout=self.settings.API_TIMEOUT,
            max_retries=self.settings.API_MAX_RETRIES
        )
        self.model = self.settings.OPENAI_MODEL

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None] | str:
        """
        与模型进行对话

        Args:
            messages: 对话历史记录列表
            stream: 是否使用流式输出
            **kwargs: 其他API参数

        Returns:
            如果stream=True，返回生成器，产生文本片段
            如果stream=False，返回完整的响应文本
        """
        try:
            logger.info(
                f"发送{'流式' if stream else ''}请求到LLM API，消息数: {len(messages)}")

            # 创建聊天完成请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs
            )

            if stream:
                # 处理流式响应
                def generate_chunks():
                    buffer = ""
                    try:
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                buffer += content
                                yield content
                        if buffer:
                            logger.info("成功完成流式响应")
                    except Exception as e:
                        logger.error(f"处理流式响应时出错: {str(e)}")
                        raise

                return generate_chunks()
            else:
                # 处理普通响应
                content = response.choices[0].message.content
                logger.info("成功获取完整响应")
                return content

        except Exception as e:
            logger.error(f"聊天请求失败: {str(e)}")
            raise
