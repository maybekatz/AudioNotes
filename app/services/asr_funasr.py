from funasr import AutoModel
from loguru import logger
from typing import Optional, Union
import threading

class FunASR:
    def __init__(self):
        self._model_lock = threading.Lock()  # 线程安全锁
        self._model: Optional[AutoModel] = None

    def _initialize_model(self):
        if self._model is not None:
            return

        with self._model_lock:  # 加锁防止并发初始化
            if self._model is None:  # 双重检查锁定
                try:
                    logger.info("Initializing FunASR model...")
                    self._model = AutoModel(
                        model="SenseVoiceSmall",
                        vad_model="fsmn-vad",
                        punc_model="ct-punc",
                        spk_model="cam++",
                        log_level="error",
                        hub="ms",
                        device="gpu",
                        # 新增优化参数
                        vad_kwargs={"max_single_segment_time": 60000},  # 60秒分段限制
                        punc_kwargs={"period": 5}  # 每5秒添加标点
                    )
                    logger.success("FunASR model initialized")
                except Exception as e:
                    logger.critical(f"Model initialization failed: {str(e)}")
                    raise RuntimeError("FunASR initialization failed") from e

    def _convert_ms_to_srt_time(self, milliseconds: int) -> str:
        """将毫秒转换为SRT时间格式 (HH:MM:SS,mmm)"""
        hours, rem = divmod(milliseconds, 3600000)
        minutes, rem = divmod(rem, 60000)
        seconds, ms = divmod(rem, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

    def _generate_srt_segment(self, 
                            index: int, 
                            start_ms: int, 
                            end_ms: int, 
                            text: str, 
                            speaker: Optional[str] = None) -> str:
        """生成SRT片段"""
        start_time = self._convert_ms_to_srt_time(start_ms)
        end_time = self._convert_ms_to_srt_time(end_ms)
        
        speaker_tag = f"[Speaker {speaker}] " if speaker else ""
        return (
            f"{index}\n"
            f"{start_time} --> {end_time}\n"
            f"{speaker_tag}{text.strip()}\n\n"
        )

    def transcribe(self, 
                  audio_path: str, 
                  output_format: str = "txt",
                  max_retries: int = 3) -> Union[str, None]:
        """执行语音转写
        
        Args:
            audio_path: 音频文件路径
            output_format: 输出格式 (txt/srt)
            max_retries: 最大重试次数
            
        Returns:
            转写结果字符串或None（失败时）
        """
        self._initialize_model()
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing audio: {audio_path} (attempt {attempt+1})")
                
                # 执行转写
                result = self._model.generate(
                    input=audio_path,
                    batch_size_s=300,
                    # 新增流控参数
                    chunk_size=2000,  # 优化长音频处理
                    hotword="",  # 自定义热词
                )
                
                if not result:
                    logger.warning("Empty transcription result")
                    return None
                
                # 处理不同输出格式
                if output_format.lower() == "srt":
                    srt_content = []
                    sentences = result[0].get('sentence_info', [])
                    
                    for idx, sent in enumerate(sentences, start=1):
                        segment = self._generate_srt_segment(
                            index=idx,
                            start_ms=sent['start'],
                            end_ms=sent['end'],
                            text=sent['text'],
                            speaker=sent.get('spk')
                        )
                        srt_content.append(segment)
                        
                    return "".join(srt_content)
                
                # 默认返回纯文本
                return result[0].get('text', '')
                
            except Exception as e:
                logger.error(f"Transcription failed (attempt {attempt+1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                
        return None

# 单例模式初始化
funasr = FunASR()
