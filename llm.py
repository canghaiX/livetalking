import time
import os
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

# Built-in LLM providers. Each exposes an OpenAI-compatible chat completions
# endpoint; the active one is selected with --llm_provider (default: dashscope).
LLM_PROVIDERS = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "orcarouter": {
        "api_key_env": "ORCAROUTER_API_KEY",
        "base_url": "https://api.orcarouter.ai/v1",
        "default_model": "orcarouter/auto",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-flash",
    },
}

EMOTIONS = {"neutral", "welcome", "serious", "closing"}
MAX_SEGMENTS = 8


def parse_dialog_segments(content):
    """Validate DeepSeek's structured reply and merge adjacent equal emotions."""
    payload = json.loads(content)
    source = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(source, list) or not source:
        raise ValueError("DeepSeek 返回内容缺少 segments")

    segments = []
    for item in source:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        emotion = item.get("emotion")
        if emotion not in EMOTIONS:
            emotion = "neutral"
        text = text.strip()
        if segments and segments[-1]["emotion"] == emotion:
            segments[-1]["text"] += text
        else:
            segments.append({"text": text, "emotion": emotion})

    if not segments:
        raise ValueError("DeepSeek 没有返回可播报文本")
    if len(segments) > MAX_SEGMENTS:
        tail = segments[MAX_SEGMENTS - 1:]
        segments = segments[:MAX_SEGMENTS - 1] + [{
            "text": "".join(segment["text"] for segment in tail),
            "emotion": "neutral",
        }]
    return segments


def _deepseek_segments(client, model, history, message):
    system = (
        "你是铁路科研展厅的中文讲解员和对话助手。结合上下文直接回答用户，表达准确、自然、口语化，"
        "控制在适合现场播报的长度。请按完整句子切分为最多 8 段，并为每段选择一种情绪。"
        "情绪只能是 neutral、welcome、serious、closing。普通事实讲解使用 neutral；"
        "问候和友好回应使用 welcome；安全、风险、重要技术结论使用 serious；"
        "结束语或致谢使用 closing。不要为了切换表情而过度改变情绪。"
        '必须输出 JSON，格式为 {"segments":[{"text":"...","emotion":"neutral"}]}。'
    )
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": message}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=1200,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = completion.choices[0].message.content
    return parse_dialog_segments(content)


def _speak_deepseek(avatar_session, segments, datainfo):
    from server.avatar_package_manager import package_manager

    for segment in segments:
        avatar_id = package_manager.emotion_avatar_id(segment["emotion"])
        prepare = getattr(avatar_session, "prepare_avatar_action", None)
        if prepare:
            try:
                prepare(avatar_id)
            except (ValueError, FileNotFoundError, OSError):
                avatar_id = avatar_session.opt.avatar_id
        event = dict(datainfo)
        event.update(emotion=segment["emotion"], avatar_action=avatar_id)
        avatar_session.put_msg_txt(segment["text"], event)


def _llm_response_deepseek(message, avatar_session, datainfo):
    with avatar_session.llm_dialog_lock:
        start = time.perf_counter()
        client = _llm_client(avatar_session.opt)
        model = _llm_model(avatar_session.opt)
        history = avatar_session.llm_history[-12:]
        segments = _deepseek_segments(client, model, history, message)
        answer = "".join(segment["text"] for segment in segments)
        avatar_session.llm_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])
        avatar_session.llm_history = avatar_session.llm_history[-12:]
        _speak_deepseek(avatar_session, segments, datainfo)
        logger.info("DeepSeek dialog completed in %.2fs (%d segments)",
                    time.perf_counter() - start, len(segments))
        return answer


def _llm_provider(opt) -> str:
    """Return the configured provider name, defaulting to dashscope."""
    return getattr(opt, 'llm_provider', 'dashscope') or 'dashscope'


def _llm_client(opt):
    """Create the OpenAI-compatible client for the configured provider."""
    from openai import OpenAI
    cfg = LLM_PROVIDERS.get(_llm_provider(opt), LLM_PROVIDERS['dashscope'])
    http_client = None
    if _llm_provider(opt) == 'deepseek':
        import httpx
        verify_tls = os.getenv('DEEPSEEK_TLS_VERIFY', '1').lower() not in ('0', 'false', 'no')
        http_client = httpx.Client(verify=verify_tls, timeout=60.0)
    return OpenAI(
        api_key=os.getenv(cfg['api_key_env']),
        base_url=cfg['base_url'],
        http_client=http_client,
    )


def _llm_model(opt) -> str:
    """Resolve the model name, falling back to the provider default."""
    cfg = LLM_PROVIDERS.get(_llm_provider(opt), LLM_PROVIDERS['dashscope'])
    return getattr(opt, 'llm_model', '') or cfg['default_model']


def llm_response(message,avatar_session:'BaseAvatar',datainfo:dict={}):
    try:
        opt = avatar_session.opt
        if _llm_provider(opt) == "deepseek":
            return _llm_response_deepseek(message, avatar_session, datainfo)
        start = time.perf_counter()
        client = _llm_client(opt)
        model = _llm_model(opt)
        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,{message}")
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': '你是一个知识助手，尽量以简短、口语化的方式输出'},
                    {'role': 'user', 'content': message}],
            stream=True,
            # Display token usage in the last line of the streamed response.
            stream_options={"include_usage": True}
        )
        result=""
        full_response=[]
        first = True
        for chunk in completion:
            if len(chunk.choices)>0:
                #print(chunk.choices[0].delta.content)
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                full_response.append(msg)
                lastpos=0
                #msglist = re.split('[,.!;:，。！?]',msg)
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；" :
                        result = result+msg[lastpos:i+1]
                        lastpos = i+1
                        if len(result)>10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result,datainfo)
                            result=""
                result = result+msg[lastpos:]
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result,datainfo)
        return "".join(full_response)

    except Exception as e:
        logger.exception('llm exceptiopn:')
        return
