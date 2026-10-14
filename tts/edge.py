import time
import asyncio
import os
import subprocess
import threading
import wave
import numpy as np
import resampy
import soundfile as sf
import edge_tts
from io import BytesIO
from pathlib import Path

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

_PIPER_VOICE = None
_PIPER_LOCK = threading.Lock()


@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        text,textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-XiaoxiaoNeural"
        voicename = textevent.get('tts', {}).get('ref_file',voice) #self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.warning('Edge TTS returned no audio; using local Mandarin voice')
            self._local_tts(text, textevent)
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint={}
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text}
            eventpoint.update(**textevent) #eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def _local_tts(self, text, textevent):
        model_path = Path(os.getenv(
            'LIVETALKING_PIPER_VOICE',
            './data/tts/zh_CN-huayan-medium.onnx',
        ))
        try:
            if not model_path.is_file():
                raise FileNotFoundError(f'Piper voice model not found: {model_path}')
            voice = self._get_piper_voice(model_path)
            audio_buffer = BytesIO()
            with wave.open(audio_buffer, 'wb') as output:
                voice.synthesize_wav(text, output)
            audio, sample_rate = sf.read(BytesIO(audio_buffer.getvalue()), dtype='float32')
            logger.info('Piper Mandarin female voice synthesized the response')
        except Exception as exc:
            logger.warning('Piper TTS unavailable, using basic local voice: %s', exc)
            result = subprocess.run(
                ['espeak-ng', '-v', 'cmn', '-s', '155', '--stdout', text],
                check=True, capture_output=True, timeout=30,
            )
            audio, sample_rate = sf.read(BytesIO(result.stdout), dtype='float32')
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sample_rate != self.sample_rate:
            audio = resampy.resample(audio, sr_orig=sample_rate, sr_new=self.sample_rate)

        total = len(audio)
        if total == 0:
            raise RuntimeError('Local Mandarin TTS generated no audio')
        for start in range(0, total, self.chunk):
            frame = audio[start:start + self.chunk]
            last = start + self.chunk >= total
            if len(frame) < self.chunk:
                frame = np.pad(frame, (0, self.chunk - len(frame)))
            event = dict(textevent)
            if start == 0:
                event.update(status='start', text=text)
            if last:
                event.update(status='end', text=text)
            self.parent.put_audio_frame(frame, event)
        logger.info('Local Mandarin TTS generated %d audio frames', (total + self.chunk - 1) // self.chunk)

    @staticmethod
    def _get_piper_voice(model_path):
        global _PIPER_VOICE
        if _PIPER_VOICE is None:
            with _PIPER_LOCK:
                if _PIPER_VOICE is None:
                    from piper import PiperVoice
                    _PIPER_VOICE = PiperVoice.load(str(model_path), use_cuda=False)
        return _PIPER_VOICE

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')
