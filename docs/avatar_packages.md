# 多动作角色包

页面：`http://222.30.145.22:8020/avatar-package.html`。仅支持当前 Wav2Lip 256 模型。

每个角色包保存同一人物的不同动作。各动作独立生成 `full_imgs`、`face_imgs` 和 `coords.pkl`，使用同一个口型模型，不重新训练。页面预览通过关闭旧 WebRTC 会话、向 `/offer` 提交新 `avatar` 并重新连接实现；对话播报时则可在同一个会话内按句子自动调度已发布动作。

动作标签除 `none` 外支持 `guide`、`wave` 和 `emphasis`：`guide` 使用03中性讲解，仅用于明确引导观看屏幕或介绍展品；`wave` 和 `emphasis` 使用统一构图、背景及前景光照后的 `gentle_introduce_07_uniform_v3`，缺失时回退 `gentle_introduce_07_facefix_v2`、`gentle_introduce_07_bgfix` 和原 `gentle_introduce_07`。旧标签 `introduce` 会兼容映射为 `guide`，普通讲解默认使用统一后的01微笑底态。02、05、06及其修正版已停用，不参与自动调度或回退，历史文件和独立预览保留。

## DeepSeek 对话和动作调度

Docker 部署从 `deployment/config.yaml` 读取 `llm_provider: deepseek` 和 `llm_model: deepseek-flash`。密钥只放在本机忽略的 `deployment/secrets.env`，由 `deployment/start.sh` 注入。首页选择“Chat LLM”后，`/human` 会结合当前会话历史生成中文回答；模型按完整句子返回 `neutral`、`welcome`、`serious`、`closing` 四种受限情绪，以及独立的 `gesture: none/guide/wave/emphasis`。02 严肃素材当前停用，`serious` 保留为语义标签但使用 01 底态。普通总结使用普通情绪，只有明确告别或致谢才使用 `closing`。旧回复缺少手势字段时按 `none` 处理。

动作标签跟随TTS音频帧进入Wav2Lip推理队列，在同一WebRTC会话内切换。切换使用按需帧解码和有限缓存，动作切换过渡为6帧。未知标签、动作缺失回退到01微笑；对话历史只保留最近六轮消息。LLM调用失败时接口报告错误，不生成替代回答。

### 表情和手势播放

`avatar_motion_config: deployment/avatar_motion.json` 启用当前 `railway_presenter_v2` 的播放配置，不修改发布 manifest 或重新生成素材。该 JSON 持久化挂载，修改后重启服务生效。更换人物时需重新选择对应素材的安全帧范围，不能直接套用本角色范围。

| 播放状态 | 原动作 | 帧范围（含两端，0 起始） | 播放方式 |
|---|---|---|---|
| 默认待机 | welcome_smile_uniform_v3 | 0–100 | 以01稳定站姿为基准统一背景、光影和色调；无语音时往返播放；回退 `welcome_smile_facefix_v2`、`welcome_smile` |
| 日常微笑/待机 | welcome_smile_uniform_v3 | 0–100 | 01作为待机和普通讲解底态，往返循环；回退 `welcome_smile_facefix_v2`、`welcome_smile` |
| 明确介绍引导 | neutral_explain_uniform_v3 | 0–164 | 03压缩为165帧（约6.6秒），抬手更快进入，完整正序一次后恢复01；回退 `neutral_explain_facefix_v2`、`neutral_explain` |
| 温和介绍/重点强调 | gentle_introduce_07_uniform_v3 | 0–149 | 07统一构图、浅灰背景和前景光照后完整正序一次，保留进入和收手，随后恢复01；依次回退 `gentle_introduce_07_facefix_v2`、`gentle_introduce_07_bgfix`、原07 |
| 告别致意 | closing_thanks_uniform_v3 | 0–249 | 04统一构图、背景和光影，最后一段音频结束后正序一次，随后恢复微笑；回退 `closing_thanks_facefix_v2`、`closing_thanks` |

情绪最短保持 `expression_hold_seconds: 4`。03和07共享同轮最多三次、播放起点至少间隔8秒的限制。07可在长回答中多次出现，手势之间至少有1秒01正常讲解，且必须在新的播报段落边界才开始。WebRTC发送回调记录实际播放位置。冷却中仅保留最新手势；回答结束和打断均清除等待手势。明确结束语的04致意等待最后一段音频的结束标记后再开始，普通总结和手动文本不触发致意。

生产动作当前使用 `*_uniform_v3` 版本：以 `welcome_smile` 稳定帧建立人脸中心、尺寸、LAB 亮度/对比度/色度和浅灰背景基准，对01、03、04、07执行固定等比例对齐、背景边缘校正及前景保守光影校正。03压缩为165帧约6.6秒，抬手在前段快速进入。对应 Avatar ID 为 `railway_presenter_v2__welcome_smile_uniform_v3`、`railway_presenter_v2__neutral_explain_uniform_v3`、`railway_presenter_v2__gentle_introduce_07_uniform_v3` 和 `railway_presenter_v2__closing_thanks_uniform_v3`。旧动作目录保留用于回退。准备流程为 `python tools/prepare_uniform_avatar.py --root . --weights models/s3fd.pth`，生成使用临时目录，四个动作全部通过校验后才原子更新清单和活动配置。

Wav2Lip 的 `l/r` 均以 20ms 音频块为单位。Mel 特征及其对应的音频对随同一批次传递；打断后由提取线程重建上下文，不丢弃新回答开头。WebRTC 将同一视频帧和两块音频绑定到同一个采样位置，使用共享单调时钟及一次性三帧预缓冲。首页和角色包预览使用一个有声视频播放器承载两条轨道，避免独立播放器产生不同的播放延迟。

`idle_return_seconds: 0.6` 避免极短合成停顿造成表情闪切；连续静音后恢复 01，已经开始的手势仍会完整收手。`transition_frames: 6` 用于动作切换，且只在 Avatar 真正变化时启用。动作各有独立游标，推理、背景与贴脸共用明确帧索引。用户打断会取消手势、清空媒体缓冲并丢弃旧播报代次的标签与音频。Wav2Lip 的 200ms Mel 窗口中心通过 `lip_sync_offset_ms`（当前 100ms）映射到视频时间戳，编号音频测试应确认实际偏差不超过一帧。

口型校准可用 `python -m tools.lipsync_calibrate --audio <serena.wav> --output /tmp/lipsync-plan.json` 生成 -120ms 到 +120ms 的20ms步进对照计划。该工具不会擅自改变生产偏移；实际听感应使用同一份Serena音频测量首个嘴部运动与音频能量起点，超过300ms首句等待或仍有中文音素形状偏差时记录为Wav2Lip模型限制。MuseTalk对照由 `python -m tools.musetalk_ab_test --models models` 检查独立权重；权重未齐全时明确保持生产Wav2Lip，禁止运行时隐式下载或自动切换。

手动文本和上传音频默认微笑，不自动触发手势。显式通过 `/offer` 选择 `avatar` 时保留原始动作预览，包括角色包页面；首页自动讲解请留空 Avatar 字段。已有素材的姿态与表情仍限制最终视觉效果。

当前服务器的出口代理使用自签名证书，因此 `deployment/secrets.env` 设置了 `DEEPSEEK_TLS_VERIFY=0`。在有正规 CA 的部署环境中应删除该设置或改为 `1`。

### 讲解员语音

默认在线音色为百炼 Qwen 实时 TTS `Serena`（`qwen3-tts-flash-realtime`）。在本机忽略文件 `deployment/secrets.env` 设置 `DASHSCOPE_API_KEY` 并重建容器启用；不要把密钥放进配置 YAML、日志或 Git。缺少密钥或在线服务暂时不可用时，先尝试 EdgeTTS `zh-CN-XiaoxiaoNeural`，再使用本地 Piper `zh_CN-huayan-medium`。当前网络访问 EdgeTTS 返回 HTTP 403，因此未配置百炼密钥时实际会使用 Piper。模型保存在被 Git 忽略的 `data/tts/` 中，并随容器挂载。`deployment/start.sh` 在首次启动时会下载并校验 Piper Voice 模型。

所有 TTS 文本在公共队列入口做中文数字归一化：年份、电话号码、12306、车次与明确编号逐位朗读；普通数值、日期、小数和百分数按中文数词朗读。播报事件携带原文用于字幕，不影响网页文字和 LLM 对话记录。转换规则见 `tts/text_normalizer.py`。

## 素材要求

- MP4 / H.264、恒定 25fps、3–15 秒、单文件不超过 100MB。
- 尺寸下限：两边至少 720px；最长边不超过 1920px，总像素不超过 1920×1080；同一角色包内所有动作尺寸一致。
- 逐帧检查单张人脸、脸宽至少 160px、检测覆盖率至少 98%。不接受多人脸、连续三帧漏检；允许的极短漏检用相邻检测框修复，并在报告中列出帧号。覆盖率不是画面审美或姿态质量保证。
- 人物身份、服装、背景、光照、机位、头手是否完整、遮脸、口部自然闭合由人工确认；当前检测器不会自动判断这些条件。推荐每段 5–8 秒、动作始终接近正面、起止回到中性站姿。需人工检查往返播放的动作是否自然。
- 该流程严格校验，不自动转换编码或帧率。可用 FFmpeg 先导出符合要求的视频；导出不能修复换脸、模糊或遮挡问题。
- 原视频音频不参与生成；上传音频可通过 `/humanaudio` 驱动口型。LiveTalking 水印保留。

## 网页工作流

创建角色包，编辑动作条目（必须含 `neutral_explain`），逐个上传视频并勾选画面确认。任务在现有单线程生成队列中依次执行；验证和解析状态持久化。失败动作可重新上传，已完成或已发布动作禁止覆盖，需要更新时创建新角色 ID。

每个已完成动作可预览原视频或连接数字人并上传测试音频。全部动作完成后可“发布并重启”：发布窗口确认后优先设置微笑为默认 Avatar（无微笑素材时使用中性讲解），当前连接断开，容器约 2 秒后由 Docker 自动重启。其他动作按需加载，不把所有片段常驻内存。页面会自动重连后台状态，不会自动恢复 WebRTC。

## API

所有角色包 API 返回 `{code, msg, data}`；失败返回相应 HTTP 错误码及原因。

| 方法和路径 | 请求 / 返回 |
|---|---|
| `POST /api/avatar/package` | JSON：`package_id`、`name`、可选 `actions:[{id,label}]`；返回持久化角色包 |
| `GET /api/avatar/packages` | 返回 `packages`、`restart_supported`、`detector_ready` |
| `GET /api/avatar/package/{id}` | 返回角色包、逐动作进度、校验报告和错误 |
| `POST /api/avatar/package/{id}/action?action_id=…&visual_confirmed=true` | multipart 唯一字段 `video_file`；返回 `task_id`。验证在后台队列执行 |
| `GET /api/avatar/package/{id}/source/{action_id}` | 支持浏览器视频播放的原始 MP4 |
| `POST /api/avatar/package/{id}/publish` | 全部动作完成且生成队列空闲时发布并重启 |

角色与动作 ID 仅允许字母、数字、下划线、短横线，最长 64 位，禁止双下划线；派生 Avatar ID 使用 `<package_id>__<action_id>`。动作数 1–8 个。

## 部署与存储

`data/avatar_packages/<id>/package.json` 保存草稿和进度；`manifest.json` 保存发布清单；`source/<action_id>.mp4` 保存上传视频。`data/avatar_packages/active.json` 保存当前活动角色。生成结果在 `data/avatars/<id>__<action_id>`。

生成先写临时目录，完整导出后原子提交；服务重启时未完成任务被标记失败，可重新上传。发布不覆盖旧 Avatar，原有单视频页面与现有 Avatar 继续可用。

`deployment/start.sh` 挂载角色包与 Avatar 目录，并设置 `LIVETALKING_ALLOW_RESTART=1`。应用不使用 Docker socket；发布响应后退出主进程，由 `--restart unless-stopped` 重启并重新读取活动角色。非 Docker 模式未开启该变量时拒绝发布重启。

S3FD 权重路径通过 `LIVETALKING_SFD_PATH` 指定，默认 `models/s3fd.pth`。本部署将检测权重打包入镜像，缺失时页面禁用上传，后台给出明确错误。权重 SHA256：`619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543`。

保持现有 8020 端口和 GPU 0。基础模型或 TTS 设置没有更改；EdgeTTS 的外部服务限制不会影响上传音频测试。
