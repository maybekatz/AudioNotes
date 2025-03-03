import os
from loguru import logger
from openai import OpenAI

async def chat_with_OPENAI(messages: list[dict], callback=None):
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o')
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.debug(f"chat with OPENAI: {base_url}, model: {model}")
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    response = client.chat.completions.create(
        model=model,
        stream=True,
        temperature=0.1,
        messages=messages
    )
    
    full_content = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:  # 检查content是否为None或空字符串
            if callback:
                await callback(content)
            full_content += content
    
    return full_content
