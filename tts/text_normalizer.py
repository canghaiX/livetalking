import re

import cn2an


_DIGITS = "零一二三四五六七八九"
_YEAR = re.compile(r"(?<!\d)(?:1\d{3}|20\d{2}|21\d{2})(?=年)")
_YEAR_RANGE = re.compile(
    r"(?<!\d)(\d{4})年?\s*[-—~～至到]\s*(\d{4})(年(?:代|间)?)?(?!\d)"
)
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_LABELED_ID = re.compile(
    r"(?:编号|代码|工号|热线|客服电话|电话|车次|序号|账号|账户|线路号|车牌号|身份证号码?|身份证号)"
    r"\s*[:：]?\s*\d{2,}(?:[- ]\d{2,})*"
)
_TRAIN_NUMBER = re.compile(r"(?<![A-Za-z0-9])[GCDZTKS]\d{1,5}(?:次)?(?![A-Za-z0-9])", re.I)
_PHONE_NUMBER = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_ID_NUMBER = re.compile(r"(?<!\d)\d{17}[\dXx](?![A-Za-z0-9])")
_HOTLINE = re.compile(
    r"(?<!\d)12306(?!\d)(?!\s*(?:人|个|元|公里|千米|米|吨|公斤|千克|台|条|座|辆|年|月|日|小时|分钟|秒|%))"
)


def _digits_to_words(value: str) -> str:
    return "".join(_DIGITS[int(char)] if char.isdigit() else char for char in value)


def normalize_for_speech(text: str) -> str:
    """Convert Arabic numerals to context-appropriate spoken Chinese."""
    if not isinstance(text, str) or not text:
        return text

    protected = {}

    def protect(match, value=None):
        original = value if value is not None else match.group()
        token = chr(0xF0000 + len(protected))
        protected[token] = original
        return token

    text = _LABELED_ID.sub(
        lambda match: protect(match, _digits_to_words(match.group())), text
    )
    text = _TRAIN_NUMBER.sub(
        lambda match: protect(match, _digits_to_words(match.group())), text
    )
    text = _PHONE_NUMBER.sub(
        lambda match: protect(match, _digits_to_words(match.group())), text
    )
    text = _ID_NUMBER.sub(
        lambda match: protect(match, _digits_to_words(match.group())), text
    )
    text = _HOTLINE.sub(lambda match: protect(match, _digits_to_words(match.group())), text)
    text = _YEAR_RANGE.sub(
        lambda match: (
            f"{_digits_to_words(match.group(1))}年到"
            f"{_digits_to_words(match.group(2))}{match.group(3) or '年'}"
        ),
        text,
    )
    text = _NUMERIC_DATE.sub(
        lambda match: (
            f"{_digits_to_words(match.group(1))}年"
            f"{cn2an.an2cn(str(int(match.group(2))), 'low')}月"
            f"{cn2an.an2cn(str(int(match.group(3))), 'low')}日"
        ),
        text,
    )
    text = _YEAR.sub(lambda match: _digits_to_words(match.group()), text)

    try:
        text = cn2an.transform(text, "an2cn")
    except (ValueError, TypeError):
        pass

    for token, original in protected.items():
        text = text.replace(token, original)
    return text


def prepare_speech_message(text: str, datainfo: dict):
    speech_event = dict(datainfo or {})
    speech_event['_tts_display_text'] = text
    return normalize_for_speech(text), speech_event


def restore_display_text(datainfo: dict) -> dict:
    event = dict(datainfo or {})
    display_text = event.pop('_tts_display_text', None)
    if display_text is not None and 'text' in event:
        event['text'] = display_text
    return event
