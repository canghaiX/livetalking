import json
import tempfile
import unittest

from llm import parse_dialog_segments
from server.avatar_package_manager import PackageManager


class DialogParsingTests(unittest.TestCase):
    def test_merges_adjacent_emotions_and_normalizes_unknown(self):
        payload = json.dumps({
            "segments": [
                {"text": "您好，", "emotion": "welcome"},
                {"text": "欢迎参观。", "emotion": "welcome"},
                {"text": "请注意安全。", "emotion": "unexpected"},
            ]
        })
        self.assertEqual(parse_dialog_segments(payload), [
            {"text": "您好，欢迎参观。", "emotion": "welcome"},
            {"text": "请注意安全。", "emotion": "neutral"},
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
            self.assertEqual(manager.emotion_avatar_id("serious"), "railway__serious_emphasis")
            self.assertEqual(manager.emotion_avatar_id("closing"), "railway__neutral_explain")


if __name__ == "__main__":
    unittest.main()
