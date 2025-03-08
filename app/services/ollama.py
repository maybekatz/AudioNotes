import os
from loguru import logger
from openai import AsyncOpenAI
from typing import Callable, Awaitable, Optional

async def chat_with_deepseek(
    messages: list[dict],
    callback: Optional[Callable[[str], Awaitable[None]]] = None
) -> str:
    """与DeepSeek API交互的异步函数
    
    Args:
        messages: 对话消息列表
        callback: 异步回调函数，用于流式输出
        
    Returns:
        str: 完整的响应内容
    """
    base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    api_key = os.getenv('DEEPSEEK_API_KEY')
    
    if not api_key:
        logger.error("DEEPSEEK_API_KEY未设置")
        raise ValueError("必须设置DEEPSEEK_API_KEY环境变量")

    logger.debug(f"初始化DeepSeek客户端 | BaseURL: {base_url} | Model: {model}")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )

    full_content = ""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=2000
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content or ""
                
                if content:
                    full_content += content
                    if callback:
                        await callback(content)

    except Exception as e:
        logger.error(f"API请求异常: {str(e)}")
        raise RuntimeError(f"DeepSeek服务调用失败: {str(e)}") from e

    finally:
        await client.close()

    logger.info(f"DeepSeek响应完成 | 总长度: {len(full_content)}字符")
    return full_content
