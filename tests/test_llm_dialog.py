import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm import (_iter_deepseek_segments, _speak_deepseek,
                 parse_dialog_segments)
from server.avatar_package_manager import PackageManager


class DialogParsingTests(unittest.TestCase):
    def test_stream_parser_yields_complete_segments_and_flushes_tail(self):
        class Delta:
            def __init__(self, content):
                self.content = content

        class Chunk:
            def __init__(self, content):
                self.choices = [SimpleNamespace(delta=Delta(content))]

        class Completions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                payload = json.dumps({'segments': [
                    {'text': '第一句。', 'emotion': 'neutral', 'gesture': 'none'},
                    {'text': '第二句没有标点', 'emotion': 'welcome', 'gesture': 'none'},
                ]}, ensure_ascii=False)
                return [Chunk(payload[:18]), Chunk(payload[18:39]), Chunk(payload[39:])]

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        result = list(_iter_deepseek_segments(client, 'deepseek-flash', [], '介绍'))
        self.assertEqual(''.join(item['text'] for item in result), '第一句。第二句没有标点')
        self.assertTrue(client.chat.completions.kwargs['stream'])
    def test_merges_adjacent_emotions_and_normalizes_unknown(self):
        payload = json.dumps({
            "segments": [
                {"text": "您好，", "emotion": "welcome"},
                {"text": "欢迎参观。", "emotion": "welcome"},
                {"text": "请注意安全。", "emotion": "unexpected"},
            ]
        })
        self.assertEqual(parse_dialog_segments(payload), [
            {"text": "您好，欢迎参观。", "emotion": "welcome", "gesture": "none"},
            {"text": "请注意安全。", "emotion": "neutral", "gesture": "none"},
        ])

    def test_limits_segments_without_losing_text(self):
        payload = json.dumps({
            "segments": [{"text": str(i), "emotion": "neutral" if i % 2 else "serious"}
                         for i in range(10)]
        })
        result = parse_dialog_segments(payload)
        self.assertEqual(len(result), 8)
        self.assertEqual("".join(item["text"] for item in result), "0123456789")

    def test_rejects_missing_segments(self):
        with self.assertRaises(ValueError):
            parse_dialog_segments(json.dumps({"answer": "没有结构化结果"}))

    def test_preserves_gesture_position_and_unknown_defaults(self):
        result = parse_dialog_segments(json.dumps({'segments': [
            {'text': '普通介绍。', 'emotion': 'neutral'},
            {'text': '请看这项成果。', 'emotion': 'neutral', 'gesture': 'introduce'},
            {'text': '继续说明。', 'emotion': 'neutral', 'gesture': 'unexpected'},
        ]}))
        self.assertEqual(result, [
            {'text': '普通介绍。', 'emotion': 'neutral', 'gesture': 'none'},
            {'text': '请看这项成果。', 'emotion': 'neutral', 'gesture': 'guide'},
            {'text': '继续说明。', 'emotion': 'neutral', 'gesture': 'none'},
        ])

    def test_accepts_emphasis_gesture_without_merging_it_away(self):
        result = parse_dialog_segments(json.dumps({'segments': [
            {'text': '这是重点成果。', 'emotion': 'serious', 'gesture': 'emphasis'},
            {'text': '后续继续说明。', 'emotion': 'serious', 'gesture': 'none'},
        ]}))
        self.assertEqual(result, [
            {'text': '这是重点成果。', 'emotion': 'serious', 'gesture': 'emphasis'},
            {'text': '后续继续说明。', 'emotion': 'serious', 'gesture': 'none'},
        ])

    def test_accepts_wave_and_renames_legacy_introduce(self):
        result = parse_dialog_segments(json.dumps({'segments': [
            {'text': '请看左侧展品。', 'emotion': 'neutral', 'gesture': 'introduce'},
            {'text': '这里是重点成果。', 'emotion': 'serious', 'gesture': 'wave'},
        ]}))
        self.assertEqual([item['gesture'] for item in result], ['guide', 'wave'])

    def test_non_string_labels_use_safe_defaults(self):
        result = parse_dialog_segments(json.dumps({'segments': [
            {'text': '正常讲解。', 'emotion': ['serious'], 'gesture': {'type': 'introduce'}},
        ]}))
        self.assertEqual(result[0]['emotion'], 'neutral')
        self.assertEqual(result[0]['gesture'], 'none')


class EmotionMappingTests(unittest.TestCase):
    def test_maps_emotions_to_completed_actions_and_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = PackageManager(folder)
            package = manager.create("railway", "讲解员", [
                {"id": "neutral_explain", "label": "中性"},
                {"id": "welcome_smile", "label": "欢迎"},
                {"id": "serious_emphasis", "label": "严肃"},
            ])
            for action in package["actions"]:
                action["status"] = "completed"
            manager._save(package)
            (manager.root / "active.json").write_text(
                json.dumps({"package_id": "railway", "avatar_id": "railway__neutral_explain"}),
                encoding="utf-8",
            )
            package["published"] = True
            manager._save(package)
            self.assertEqual(manager.emotion_avatar_id("welcome"), "railway__welcome_smile")
            self.assertEqual(manager.emotion_avatar_id("neutral"), "railway__welcome_smile")
            self.assertEqual(manager.emotion_avatar_id("serious"), "railway__welcome_smile")
            self.assertEqual(manager.emotion_avatar_id("closing"), "railway__welcome_smile")

    def test_closing_marker_is_attached_only_to_final_segment(self):
        events = []
        avatar = SimpleNamespace(
            motion_scheduler=object(),
            put_msg_txt=lambda text, event: events.append((text, event)),
        )
        segments = [
            {"text": "讲解重点。", "emotion": "serious", "gesture": "none"},
            {"text": "感谢参观，再见。", "emotion": "closing", "gesture": "none"},
        ]
        with patch('server.avatar_package_manager.package_manager.emotion_avatar_id',
                   return_value='railway__welcome_smile'):
            _speak_deepseek(avatar, segments, {'_speech_epoch': 0})
        self.assertNotIn('_close_after_audio', events[0][1])
        self.assertFalse(events[0][1]['_round_final'])
        self.assertTrue(events[1][1]['_close_after_audio'])
        self.assertTrue(events[1][1]['_round_final'])


if __name__ == "__main__":
    unittest.main()
