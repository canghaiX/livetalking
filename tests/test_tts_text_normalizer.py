import unittest
from queue import Queue
from unittest.mock import Mock

from tts.base_tts import BaseTTS
from tts.text_normalizer import (
    normalize_for_speech,
    prepare_speech_message,
    restore_display_text,
)


class SpokenTextTests(unittest.TestCase):
    def test_years_hotline_identifiers_and_phone_numbers(self):
        self.assertEqual(normalize_for_speech('成立于1950年，客服电话12306。'),
                         '成立于一九五零年，客服电话一二三零六。')
        self.assertEqual(normalize_for_speech('电话13800138000，车次G123。'),
                         '电话一三八零零一三八零零零，车次G一二三。')
        self.assertEqual(normalize_for_speech('线路编号00123。'), '线路编号零零一二三。')

    def test_cardinals_dates_decimals_percentages_and_mixed_text(self):
        self.assertEqual(normalize_for_speech('线路长12306公里。'), '线路长一万二千三百零六公里。')
        self.assertEqual(normalize_for_speech('日期是2024年5月1日。'), '日期是二零二四年五月一日。')
        self.assertEqual(normalize_for_speech('日期是2024-05-01。'), '日期是二零二四年五月一日。')
        self.assertEqual(normalize_for_speech('自1950年至2020年建设。'),
                         '自一九五零年到二零二零年建设。')
        self.assertEqual(normalize_for_speech('提升了12.5%，覆盖3条线路。'),
                         '提升了百分之十二点五，覆盖三条线路。')
        self.assertEqual(normalize_for_speech('G123次列车运行350.5公里。'),
                         'G一二三次列车运行三百五十点五公里。')

    def test_audio_events_keep_original_subtitle_without_mutation(self):
        original = {'emotion': 'neutral'}
        speech_text, speech_event = prepare_speech_message('建成于1950年。', original)

        self.assertEqual(speech_text, '建成于一九五零年。')
        self.assertEqual(speech_event['_tts_display_text'], '建成于1950年。')
        self.assertEqual(original, {'emotion': 'neutral'})
        self.assertEqual(
            restore_display_text({'status': 'start', 'text': speech_text,
                                  **speech_event}),
            {'status': 'start', 'text': '建成于1950年。', 'emotion': 'neutral'},
        )

    def test_shared_tts_queue_normalizes_manual_and_llm_messages(self):
        tts = BaseTTS.__new__(BaseTTS)
        tts.msgqueue = Queue()
        tts.state = None
        tts.put_msg_txt('客服电话12306，建成于1950年。', {'emotion': 'neutral'})
        tts.txt_to_audio = Mock()
        tts.stop_tts = Mock()

        class StopAfterOneMessage:
            calls = 0

            def is_set(self):
                self.calls += 1
                return self.calls > 1

        tts.process_tts(StopAfterOneMessage())

        speech_text, event = tts.txt_to_audio.call_args.args[0]
        self.assertEqual(speech_text, '客服电话一二三零六，建成于一九五零年。')
        self.assertEqual(event['_tts_display_text'], '客服电话12306，建成于1950年。')
        self.assertEqual(event['emotion'], 'neutral')


if __name__ == '__main__':
    unittest.main()
