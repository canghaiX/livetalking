###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle
import json
from collections import OrderedDict

import queue
from queue import Queue
from threading import Event, RLock, Thread
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar
from avatars.motion_scheduler import MotionScheduler

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))


def blend_wav2lip_face(source_face, predicted_face, start_ratio=0.44,
                       feather_ratio=0.10):
    """Keep the source eyes/eyebrows and blend Wav2Lip only into the mouth area."""
    source = np.asarray(source_face, dtype=np.float32)
    predicted = np.asarray(predicted_face, dtype=np.float32)
    if source.ndim != 3 or predicted.ndim != 3:
        raise ValueError('face crops must be HxWxC images')
    if predicted.shape[:2] != source.shape[:2]:
        predicted = cv2.resize(predicted, (source.shape[1], source.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    height = source.shape[0]
    # ``start_ratio`` is the point at which blending begins; ``feather_ratio``
    # is the full transition width.  Thus the default protects 0-44% and
    # reaches the generated mouth by 54% of the crop height.
    lower = float(np.clip(start_ratio, 0.0, 1.0)) * height
    feather = max(1.0, float(np.clip(feather_ratio, 0.0, 1.0)) * height)
    upper = lower + feather
    rows = np.arange(height, dtype=np.float32)
    alpha = np.clip((rows - lower) / max(1.0, upper - lower), 0.0, 1.0)
    # Smoothstep avoids a visible horizontal seam at the preserved-eye edge.
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    alpha = alpha[:, None, None]
    blended = source * (1.0 - alpha) + predicted * alpha
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

class LazyFrameCycle:
    """Decode full-resolution frames on demand so emotion actions use bounded RAM."""
    def __init__(self, paths, cache_size=12):
        self.paths = paths
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index %= len(self.paths)
        frame = self.cache.pop(index, None)
        if frame is None:
            frame = cv2.imread(self.paths[index])
            if frame is None:
                raise ValueError(f"Unable to read avatar frame: {self.paths[index]}")
        self.cache[index] = frame
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame


def load_avatar(avatar_id, lazy_full=False):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = LazyFrameCycle(input_img_list) if lazy_full else read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar
        self.default_avatar_payload = avatar
        self._action_avatar_cache = OrderedDict()
        self._action_avatar_lock = RLock()
        self.face_blend_start_ratio = float(getattr(opt, 'face_blend_start_ratio', 0.44))
        self.face_blend_feather_ratio = float(getattr(opt, 'face_blend_feather_ratio', 0.10))
        self._configure_motion()

        self.asr = MelASR(opt,self)
        self.asr.warm_up()

    def _configure_motion(self):
        path = getattr(self.opt, 'avatar_motion_config', '')
        if not path or getattr(self.opt, 'avatar_motion_preview', False):
            return
        from server.avatar_package_manager import package_manager
        try:
            with open(path, encoding='utf-8') as stream:
                config = json.load(stream)
            package_id = config['package_id']
            if self.opt.avatar_id != f"{package_id}__{config['profiles']['smile']['action_id']}":
                return
            fallback_actions = {
                # Versioned uniform-v3 actions are preferred; older generated
                # assets remain available as explicit rollback paths.
                'idle': ['welcome_smile_facefix_v2', 'welcome_smile'],
                'smile': ['welcome_smile_facefix_v2', 'welcome_smile'],
                'serious': ['welcome_smile_facefix_v2', 'welcome_smile'],
                'guide': ['neutral_explain_facefix_v2', 'neutral_explain'],
                'wave': ['gentle_introduce_07_facefix_v2',
                         'gentle_introduce_07_bgfix', 'gentle_introduce_07'],
                'emphasis': ['gentle_introduce_07_facefix_v2',
                             'gentle_introduce_07_bgfix', 'gentle_introduce_07'],
                'closing': ['closing_thanks_facefix_v2', 'closing_thanks'],
                'nod': ['closing_thanks_facefix_v2', 'closing_thanks'],
            }
            profiles = {}
            for name, profile in config['profiles'].items():
                profile = dict(profile)
                desired_action = profile['action_id']
                desired_avatar = f"{package_id}__{desired_action}"
                if package_manager.is_active_avatar(desired_avatar):
                    profile['avatar_id'] = desired_avatar
                    profiles[name] = profile
                    continue
                fallback_list = fallback_actions.get(name, ['welcome_smile'])
                fallback_action = next((candidate for candidate in fallback_list
                                        if package_manager.is_active_avatar(f'{package_id}__{candidate}')),
                                       None)
                if fallback_action:
                    fallback_avatar = f"{package_id}__{fallback_action}"
                    profile['action_id'] = fallback_action
                    profile['avatar_id'] = fallback_avatar
                    profiles[name] = profile
                    logger.warning('Motion profile %s unavailable; falling back to %s',
                                   name, fallback_action)
                else:
                    logger.warning('Motion profile %s and fallback %s unavailable',
                                   name, fallback_action)
            config['profiles'] = profiles
            if 'smile' not in profiles:
                raise ValueError('No usable smile motion profile is available')
            scheduler = MotionScheduler(config, self.opt.fps)
            if profiles['smile']['end'] >= self.get_avatar_length():
                raise ValueError('Smile playback range exceeds available frames')
            self.motion_scheduler = scheduler
            logger.info('Motion scheduling enabled: 01 baseline, guide/wave/emphasis gestures, cooldown=%ss',
                        scheduler.cooldown)
        except (OSError, KeyError, TypeError, ValueError):
            logger.exception('Unable to configure motion playback; keeping original avatar preview')

    def infer_motion_batch(self, features, audio_frames, timing=None):
        """Select once per video frame, then batch only identical avatar payloads."""
        selected = []
        scheduler = self.motion_scheduler
        for offset in range(min(len(features), self.batch_size)):
            pair = audio_frames[offset*2:offset*2+2]
            pair = [frame if frame.userdata.get('_speech_epoch', self.speech_epoch) == self.speech_epoch
                    else type(frame)(np.zeros_like(frame.data), 1, {}) for frame in pair]
            speaking = any(frame.type == 0 for frame in pair)
            event = next((frame.userdata for frame in pair if frame.userdata.get('_utterance_id')), {})
            choice = scheduler.step(event, speaking,
                                    audio_events=[frame.userdata for frame in pair])
            profile = scheduler.profiles[choice.profile]
            try:
                payload = self.avatar_payload_for(profile['avatar_id'])
                if profile['end'] >= min(len(part) for part in payload):
                    raise ValueError('Playback range exceeds avatar frame count')
            except (ValueError, FileNotFoundError, OSError) as exc:
                logger.warning('Motion profile %s failed; using smile: %s', choice.profile, exc)
                scheduler.disable(choice.profile)
                choice = scheduler.step({}, speaking)
                payload = self.default_avatar_payload
                profile = scheduler.profiles[choice.profile]
            frame_timing = {}
            if timing is not None and offset < len(timing.mel_sample_starts):
                pair_start = pair[0].userdata.get('_audio_sample_start')
                pair_end = (pair[-1].userdata.get('_audio_sample_start', pair_start)
                            + getattr(self, 'chunk', 320))
                frame_timing = {
                    '_audio_sample_start': pair_start,
                    '_audio_sample_end': pair_end,
                    '_mel_sample_start': timing.mel_sample_starts[offset],
                    '_mel_sample_end': timing.mel_sample_ends[offset],
                    '_video_frame_index': int(timing.mel_sample_starts[offset] //
                                              (getattr(self, 'chunk', 320) * 2)),
                    '_speech_epoch': choice.epoch,
                }
            if choice.started:
                userdata = dict(pair[0].userdata)
                userdata.update(frame_timing)
                userdata.update(_motion_started=choice.started,
                                _motion_profile=choice.profile,
                                _speech_epoch=choice.epoch)
                pair[0] = type(pair[0])(pair[0].data, pair[0].type,
                                       userdata)
            elif frame_timing:
                userdata = dict(pair[0].userdata)
                userdata.update(frame_timing)
                pair[0] = type(pair[0])(pair[0].data, pair[0].type,
                                        userdata)
            selected.append((payload, choice.index, pair,
                             {'epoch': choice.epoch, 'profile': choice.profile,
                              'avatar_id': profile['avatar_id'], **frame_timing}))

        output = []
        start = 0
        while start < len(selected):
            payload = selected[start][0]
            end = start + 1
            while end < len(selected) and selected[end][0] is payload:
                end += 1
            if any(frame.type == 0 for item in selected[start:end] for frame in item[2]):
                predictions = self.inference_batch(
                    0, features[start:end], avatar_payload=payload,
                    frame_indices=[item[1] for item in selected[start:end]])
            else:
                predictions = [None] * (end - start)
            for prediction, (_, index, pair, meta) in zip(predictions, selected[start:end]):
                output.append((prediction, pair, index, payload, meta))
            start = end
        return output
    
    def prepare_avatar_action(self, avatar_id):
        from server.avatar_package_manager import package_manager

        if avatar_id == self.opt.avatar_id:
            return
        if not package_manager.is_active_avatar(avatar_id):
            raise ValueError("动作不属于当前已发布角色包")
        with self._action_avatar_lock:
            if avatar_id in self._action_avatar_cache:
                self._action_avatar_cache.move_to_end(avatar_id)
                return
            self._action_avatar_cache[avatar_id] = load_avatar(avatar_id, lazy_full=True)
            while len(self._action_avatar_cache) > 4:
                self._action_avatar_cache.popitem(last=False)

    def avatar_payload_for(self, avatar_id):
        if not avatar_id or avatar_id == self.opt.avatar_id:
            return self.default_avatar_payload
        self.prepare_avatar_action(avatar_id)
        with self._action_avatar_lock:
            self._action_avatar_cache.move_to_end(avatar_id)
            return self._action_avatar_cache[avatar_id]

    def inference_batch(self, index, audiofeat_batch, avatar_payload=None, frame_indices=None):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        face_list = avatar_payload[1] if avatar_payload else self.face_list_cycle
        length = len(face_list)
        img_batch = []
        for i in range(len(audiofeat_batch)):
            idx = frame_indices[i] if frame_indices is not None else mirror_index(length, index + i)
            face = face_list[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self,pred_frame,idx:int,avatar_payload=None):
        frame_list, face_list, coord_list = avatar_payload or self.default_avatar_payload
        bbox = coord_list[idx]
        combine_frame = frame_list[idx].copy()
        y1, y2, x1, x2 = bbox
        # Older/custom avatar payloads may not contain face crops. Keep their
        # existing behavior while all generated Wav2Lip avatars use the eye
        # protection blend above.
        if face_list is None:
            stable_face = pred_frame
        else:
            source_face = face_list[idx]
            stable_face = blend_wav2lip_face(
                source_face, pred_frame,
                start_ratio=self.face_blend_start_ratio,
                feather_ratio=self.face_blend_feather_ratio,
            )
        res_frame = cv2.resize(stable_face, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame

