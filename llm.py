import time
import os
import json
from uuid import uuid4
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
GESTURE_ALIASES = {
    "none": "none",
    "guide": "guide",
    "wave": "wave",
    "emphasis": "emphasis",
    # Replies from the previous prompt remain compatible.
    "introduce": "guide",
}
MAX_SEGMENTS = 8
SENTENCE_PUNCTUATION = set(".!;。！？；")
SOFT_PUNCTUATION = set(",:，：")


def _normalize_dialog_item(item):
    """Normalize one structured segment without merging it with its neighbours."""
    if not isinstance(item, dict):
        return None
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    emotion = item.get("emotion")
    if not isinstance(emotion, str) or emotion not in EMOTIONS:
        emotion = "neutral"
    gesture = item.get("gesture", "none")
    gesture = GESTURE_ALIASES.get(gesture, "none") if isinstance(gesture, str) else "none"
    return {"text": text.strip(), "emotion": emotion, "gesture": gesture}


def _split_stream_text(text, max_chars=40, min_soft_chars=20):
    """Split at sentence boundaries; split commas only after enough text.

    Very short comma-delimited requests create many independent realtime TTS
    responses and make a transient network delay audible.  A soft boundary is
    therefore only used after a useful amount of speech has accumulated.
    """
    text = (text or '').strip()
    if not text:
        return []
    max_chars = max(20, int(max_chars or 40))
    min_soft_chars = min(max_chars, max(12, int(min_soft_chars or 20)))
    result, current = [], []
    for char in text:
        current.append(char)
        if (char in SENTENCE_PUNCTUATION
                or (char in SOFT_PUNCTUATION and len(current) >= min_soft_chars)
                or len(current) >= max_chars):
            result.append(''.join(current).strip())
            current = []
    if current:
        result.append(''.join(current).strip())
    return [part for part in result if part]


def _stream_json_objects(raw, scan_pos=0):
    """Yield complete objects from the streamed ``segments`` JSON array.

    DeepSeek streams JSON tokens, so an individual segment can arrive over
    several chunks.  This scanner waits only for a balanced object and does
    not require the closing array or the rest of the response.
    """
    array_start = raw.find('[', raw.find('"segments"'))
    if array_start < 0:
        return [], len(raw)
    pos = max(scan_pos, array_start + 1)
    objects = []
    depth = 0
    start = None
    in_string = False
    escaped = False
    while pos < len(raw):
        char = raw[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == '{':
            if depth == 0:
                start = pos
            depth += 1
        elif char == '}':
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objects.append(json.loads(raw[start:pos + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
        pos += 1
    return objects, pos


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
        normalized = _normalize_dialog_item(item)
        if normalized is None:
            continue
        text = normalized["text"]
        emotion = normalized["emotion"]
        gesture = normalized["gesture"]
        if (segments and segments[-1]["emotion"] == emotion
                and segments[-1].get("gesture", "none") == "none" and gesture == "none"):
            segments[-1]["text"] += text
        else:
            segments.append({"text": text, "emotion": emotion, "gesture": gesture})

    if not segments:
        raise ValueError("DeepSeek 没有返回可播报文本")
    if len(segments) > MAX_SEGMENTS:
        tail = segments[MAX_SEGMENTS - 1:]
        segments = segments[:MAX_SEGMENTS - 1] + [{
            "text": "".join(segment["text"] for segment in tail),
            "emotion": "neutral",
            "gesture": "none",
        }]
    return segments


def _deepseek_prompt():
    return (
        "你是铁路科研展厅的中文讲解员和对话助手。结合上下文直接回答用户，表达准确、自然、口语化，"
        "控制在适合现场播报的长度。请按完整句子切分为最多 8 段，并为每段选择一种情绪。"
        "情绪只能是 neutral、welcome、serious、closing。普通事实讲解使用 neutral；"
        "问候和友好回应使用 welcome；安全、风险、重要技术结论使用 serious；"
        "只有明确的告别或致谢使用 closing，普通总结不用 closing。情绪与手势分别选择。"
        "手势 gesture 只能是 none、guide、wave 或 emphasis，默认 none；明确介绍展品、设备、成果或引导观看屏幕时用 guide，"
        "在自然段落停顿处需要温和介绍手势时用 wave，需要强调关键成果或结论时用 emphasis；这两种标签都播放07温和介绍动作。"
        "长回答可在自然段落开头安排少量 wave 或 emphasis，每次回答最多三段使用手势，"
        "两个手势之间应有足够讲解内容，尽量间隔至少八秒；短回答使用零到一个手势。"
        "每次手势之后至少安排一段 gesture 为 none 的正常讲解，禁止连续段落触发手势；05和06动作已经停用。"
        "普通说明不要使用手势，不要连续每句都摆手；只有明确介绍具体对象时才使用 guide。"
        "不要为了切换表情而过度改变情绪。"
        '必须输出 JSON，格式为 {"segments":[{"text":"...","emotion":"neutral","gesture":"none"}]}。'
    )


def _deepseek_messages(history, message):
    return [{"role": "system", "content": _deepseek_prompt()}, *history,
            {"role": "user", "content": message}]


def _deepseek_segments(client, model, history, message):
    messages = _deepseek_messages(history, message)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=1200,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = completion.choices[0].message.content
    return parse_dialog_segments(content)


def _iter_deepseek_segments(client, model, history, message, max_chars=40):
    """Stream complete structured segments as soon as their JSON object closes."""
    messages = _deepseek_messages(history, message)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=1200,
        extra_body={"thinking": {"type": "disabled"}},
        stream=True,
    )
    raw = ''
    yielded = 0
    for chunk in completion:
        choices = getattr(chunk, 'choices', None) or []
        if not choices:
            continue
        delta = getattr(choices[0], 'delta', None)
        content = getattr(delta, 'content', None) if delta is not None else None
        if not content:
            continue
        raw += content
        objects, _ = _stream_json_objects(raw, 0)
        for item in objects[yielded:]:
            yielded += 1
            normalized = _normalize_dialog_item(item)
            if normalized is None:
                continue
            for part in _split_stream_text(normalized['text'], max_chars):
                yield {**normalized, 'text': part}
    # Some compatible gateways buffer JSON until the end. Parse it here so the
    # last segment is never lost, while avoiding duplicates from the scanner.
    try:
        payload = json.loads(raw)
        source = payload.get('segments') if isinstance(payload, dict) else []
    except (TypeError, json.JSONDecodeError):
        source = []
    for item in source[yielded:]:
        normalized = _normalize_dialog_item(item)
        if normalized is None:
            continue
        for part in _split_stream_text(normalized['text'], max_chars):
            yield {**normalized, 'text': part}


def _speak_deepseek(avatar_session, segments, datainfo):
    from server.avatar_package_manager import package_manager

    round_id = datainfo.get('_round_id') or uuid4().hex
    generation = datainfo.get('_dialog_generation', getattr(avatar_session, 'dialog_generation', 0))
    for index, segment in enumerate(segments):
        if generation != getattr(avatar_session, 'dialog_generation', 0):
            logger.info('Discarding stale DeepSeek response generation=%s', generation)
            return False
        avatar_id = package_manager.emotion_avatar_id(segment["emotion"])
        prepare = getattr(avatar_session, "prepare_avatar_action", None)
        if prepare and not getattr(avatar_session, 'motion_scheduler', None):
            try:
                prepare(avatar_id)
            except (ValueError, FileNotFoundError, OSError):
                avatar_id = avatar_session.opt.avatar_id
        event = dict(datainfo)
        event.update(emotion=segment["emotion"], gesture=segment.get('gesture', 'none'),
                     _round_id=round_id, _segment_index=index,
                     _round_final=index == len(segments) - 1)
        if index == len(segments) - 1 and segment["emotion"] == "closing":
            event['_close_after_audio'] = True
            event['_round_final'] = True
        if not getattr(avatar_session, 'motion_scheduler', None):
            event['avatar_action'] = avatar_id
        avatar_session.put_msg_txt(segment["text"], event)
    return True


def _speak_deepseek_segment(avatar_session, segment, datainfo, final=False,
                            segment_index=0):
    """Enqueue one segment; used by the streaming DeepSeek path."""
    from server.avatar_package_manager import package_manager

    generation = datainfo.get('_dialog_generation', getattr(avatar_session, 'dialog_generation', 0))
    if generation != getattr(avatar_session, 'dialog_generation', 0):
        return False
    avatar_id = package_manager.emotion_avatar_id(segment["emotion"])
    prepare = getattr(avatar_session, "prepare_avatar_action", None)
    if prepare and not getattr(avatar_session, 'motion_scheduler', None):
        try:
            prepare(avatar_id)
        except (ValueError, FileNotFoundError, OSError):
            avatar_id = avatar_session.opt.avatar_id
    event = dict(datainfo)
    event.update(emotion=segment["emotion"], gesture=segment.get('gesture', 'none'),
                 _round_id=datainfo.get('_round_id'), _segment_index=segment_index,
                 _round_final=bool(final))
    if final and segment["emotion"] == "closing":
        event['_close_after_audio'] = True
    if not getattr(avatar_session, 'motion_scheduler', None):
        event['avatar_action'] = avatar_id
    avatar_session.put_msg_txt(segment["text"], event)
    return True


def _llm_response_deepseek(message, avatar_session, datainfo):
    start = time.perf_counter()
    generation = datainfo.get('_dialog_generation', getattr(avatar_session, 'dialog_generation', 0))
    with avatar_session.llm_dialog_lock:
        history = list(avatar_session.llm_history[-12:])
    client = _llm_client(avatar_session.opt)
    model = _llm_model(avatar_session.opt)
    max_chars = getattr(avatar_session.opt, 'llm_stream_segment_max_chars', 40)
    segments = []
    pending = None
    segment_index = 0
    stream_failed = None
    try:
        streamed = _iter_deepseek_segments(client, model, history, message, max_chars)
        for segment in streamed:
            # Keep one segment back so the final event can be marked reliably.
            # This is also the natural prefetch window for Qwen audio.
            if pending is not None:
                segments.append(pending)
                if not _speak_deepseek_segment(avatar_session, pending, datainfo,
                                               final=False, segment_index=segment_index):
                    return None
                segment_index += 1
            pending = segment
    except (TypeError, AttributeError) as exc:
        # Older OpenAI-compatible clients may not accept stream=True. Fall
        # back to one complete structured response, preserving compatibility.
        stream_failed = exc
    except Exception:
        logger.exception('DeepSeek 流式响应失败')
        stream_failed = True
    if not segments and pending is None:
        try:
            complete = _deepseek_segments(client, model, history, message)
        except Exception:
            logger.exception('DeepSeek structured response failed')
            return None
        segments = complete
        for index, segment in enumerate(segments):
            if not _speak_deepseek_segment(avatar_session, segment, datainfo,
                                           final=index == len(segments) - 1,
                                           segment_index=index):
                return None
    elif pending is not None:
        segments.append(pending)
        if not _speak_deepseek_segment(avatar_session, pending, datainfo, final=True,
                                       segment_index=segment_index):
            return None
    if not segments:
        return None
    answer = "".join(segment["text"] for segment in segments)
    if not avatar_session.is_dialog_current(generation):
        logger.info('Ignoring stale DeepSeek answer generation=%s', generation)
        return None
    with avatar_session.speech_lock:
        with avatar_session.llm_dialog_lock:
            if not avatar_session.is_dialog_current(generation):
                return None
            avatar_session.llm_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ])
            avatar_session.llm_history = avatar_session.llm_history[-12:]
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
    datainfo = dict(datainfo)
    datainfo.setdefault('_speech_epoch', getattr(avatar_session, 'speech_epoch', 0))
    datainfo.setdefault('_dialog_generation', getattr(avatar_session, 'dialog_generation', 0))
    datainfo.setdefault('_round_id', uuid4().hex)
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
