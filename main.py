import os
import uuid
import asyncio
import chainlit as cl
from io import BytesIO
from chainlit import ThreadDict
from chainlit.element import ElementBased
from loguru import logger
from app.services import data_layer
from app.services.asr_funasr import funasr
from app.services.ollama import chat_with_ollama
from app.utils import utils

# load environment variables
from dotenv import load_dotenv
load_dotenv()

# 配置日志
logger.remove()
logger.add(f"{utils.storage_dir('logs')}/log.log", rotation="500 MB")

# 初始化数据层
data_layer.init()

@cl.password_auth_callback
def password_auth_callback(username: str, password: str):
    u = os.getenv("USERNAME", "admin")
    p = os.getenv("PASSWORD", "admin")
    if (username, password) == (u, p):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def on_chat_start():
    try:
        files = None
        while files == None:
            msg = cl.AskFileMessage(
                content="请上传一个**音频/视频**文件",
                # accept=["audio/*", "video/*"],
                accept=["*/*"],
                max_size_mb=10240,
            )
            files = await msg.send()
        
        file = files[0]
        msg = cl.Message(content="")
        
        # 初始化会话状态
        cl.user_session.set("has_transcription", False)
        
        async def transcribe_file(uploaded_file):
            await msg.stream_token(f"文件 《{uploaded_file.name}》 上传成功, 识别中...\n")
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, funasr.transcribe, uploaded_file.path)
                await msg.stream_token(f"## 识别结果 \n{result}\n")
                
                # 存储识别结果到会话状态
                cl.user_session.set("transcription", result)
                cl.user_session.set("has_transcription", True)
                
                return result
            except Exception as e:
                error_msg = f"识别过程中出错: {str(e)}"
                logger.error(error_msg)
                await msg.stream_token(f"❌ {error_msg}\n")
                return None
        
        async def summarize_notes(text):
            if not text:
                await msg.stream_token("⚠️ 没有可用的识别结果，无法生成笔记。\n")
                return
                
            messages = [
                {"role": "system", "content": "你是一名笔记整理专家，根据用户提供的内容，整理出一份内容详尽的结构化的笔记"},
                {"role": "user", "content": text},
            ]
            
            async def on_message(content):
                await msg.stream_token(content)
            
            try:
                await msg.stream_token("## 整理笔记\n\n")
                summary = await chat_with_deepseek(messages, callback=on_message)  # 使用deepseek服务
                
                # 存储笔记到会话状态
                cl.user_session.set("notes_summary", summary)
                
            except Exception as e:
                error_msg = f"生成笔记时出错: {str(e)}"
                logger.error(error_msg)
                await msg.stream_token(f"❌ {error_msg}\n")
        
        asr_result = await transcribe_file(file)
        await summarize_notes(asr_result)
        await msg.send()
        
    except Exception as e:
        error_msg = f"处理过程中出现未预期的错误: {str(e)}"
        logger.error(error_msg)
        await cl.Message(content=f"❌ {error_msg}").send()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    try:
        if chunk.isStart:
            buffer = BytesIO()
            buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
            cl.user_session.set("audio_buffer", buffer)
            cl.user_session.set("audio_mime_type", chunk.mimeType)
        
        audio_buffer = cl.user_session.get("audio_buffer")
        if audio_buffer:
            audio_buffer.write(chunk.data)
    except Exception as e:
        logger.error(f"处理音频块时出错: {str(e)}")

@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    try:
        audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
        if not audio_buffer:
            await cl.Message(content="❌ 未找到音频数据").send()
            return
            
        audio_buffer.seek(0)
        file_path = f"{utils.upload_dir()}/{str(uuid.uuid4())}.wav"
        
        try:
            with open(file_path, "wb") as f:
                f.write(audio_buffer.read())
            
            # 记录临时文件路径，以便后续清理
            temp_files = cl.user_session.get("temp_files", [])
            temp_files.append(file_path)
            cl.user_session.set("temp_files", temp_files)
            
            result = funasr.transcribe(file_path)
            
            # 更新会话状态
            cl.user_session.set("transcription", result)
            cl.user_session.set("has_transcription", True)
            
            await cl.Message(
                content=result,
                type="user_message",
            ).send()
            
            await chat()
            
        except Exception as e:
            error_msg = f"处理音频文件时出错: {str(e)}"
            logger.error(error_msg)
            await cl.Message(content=f"❌ {error_msg}").send()
            
    except Exception as e:
        error_msg = f"处理音频结束事件时出错: {str(e)}"
        logger.error(error_msg)
        await cl.Message(content=f"❌ {error_msg}").send()

async def chat():
    try:
        # 获取聊天历史
        history = cl.chat_context.to_openai()
        logger.info(f"Chat history: {history}")
        
        # 检查是否有识别结果
        has_transcription = cl.user_session.get("has_transcription", False)
        if not has_transcription:
            await cl.Message(content="⚠️ 请先上传音频文件进行识别").send()
            return
            
        msg = cl.Message(content="")
        
        # 确保系统提示只添加一次
        if len(history) >= 2 and history[0]["role"] == "system":
            messages = history
        else:
            transcription = cl.user_session.get("transcription", "")
            messages = [
                {"role": "system", "content": "你是一名笔记整理专家，严格根据音频识别的结果和整理的笔记内容，回答用户的问题。"},
                {"role": "user", "content": f"这是识别的音频内容：\n\n{transcription}\n\n请基于这些内容回答我的问题。"}
            ]
            # 添加用户的当前提问
            if history and history[-1]["role"] == "user":
                messages.append(history[-1])
            
        logger.info(f"Processed messages: {messages}")
        
        async def on_message(content):
            await msg.stream_token(content)
        
        await chat_with_OPENAI(messages, callback=on_message)  # 使用OPENAI服务
        await msg.send()
        
    except Exception as e:
        error_msg = f"聊天过程中出错: {str(e)}"
        logger.error(error_msg)
        await cl.Message(content=f"❌ {error_msg}").send()

@cl.on_message
async def on_message(message: cl.Message):
    await chat()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # 可以在这里恢复会话状态
    pass

@cl.on_chat_end
async def on_chat_end():
    # 清理临时文件
    try:
        temp_files = cl.user_session.get("temp_files", [])
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"清理临时文件时出错: {str(e)}")
