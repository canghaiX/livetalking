# 多动作角色包

页面：`http://222.30.145.22:8020/avatar-package.html`。仅支持当前 Wav2Lip 256 模型。

每个角色包保存同一人物的不同动作。各动作独立生成 `full_imgs`、`face_imgs` 和 `coords.pkl`，使用同一个口型模型，不重新训练。页面预览通过关闭旧 WebRTC 会话、向 `/offer` 提交新 `avatar` 并重新连接实现；对话播报时则可在同一个会话内按句子自动调度已发布动作。

## DeepSeek 对话和动作调度

Docker 部署从 `deployment/config.yaml` 读取 `llm_provider: deepseek` 和 `llm_model: deepseek-flash`。密钥只放在本机忽略的 `deployment/secrets.env`，由 `deployment/start.sh` 注入。首页选择“Chat LLM”后，`/human` 会结合当前会话历史生成中文回答；模型按完整句子返回 `neutral`、`welcome`、`serious`、`closing` 四种受限情绪，服务端将它们映射到已发布角色包的四个动作。

动作标签跟随 TTS 音频帧进入 Wav2Lip 推理队列，在同一 WebRTC 会话内切换。切换使用按需帧解码和有限缓存，并在视频渲染端做约 240ms 的交叉淡化。未知标签、动作缺失或 API 失败会回退到默认讲解动作；对话历史只保留最近六轮消息。

当前服务器的出口代理使用自签名证书，因此 `deployment/secrets.env` 设置了 `DEEPSEEK_TLS_VERIFY=0`。在有正规 CA 的部署环境中应删除该设置或改为 `1`。

### 讲解员语音

默认在线音色为 `zh-CN-XiaoxiaoNeural`。Edge TTS 不可用时，服务使用本地 Piper `zh_CN-huayan-medium` 普通话女声；模型保存在被 Git 忽略的 `data/tts/` 中，并随容器挂载。`deployment/start.sh` 在首次启动时会下载并校验 Piper Voice 模型。模型来自 Piper Voices 的 MIT 许可模型集，SHA-256 为 `9929917bf8cabb26fd528ea44d3a6699c11e87317a14765312420be230be0f3d`。当前网络如有自签名代理证书，可在确认代理可信后用 `CURL_INSECURE=1 deployment/download_piper_voice.sh` 下载。Piper 也不可用时，才会回退到系统基础语音。

## 素材要求

- MP4 / H.264、恒定 25fps、3–15 秒、单文件不超过 100MB。
- 尺寸下限：两边至少 720px；最长边不超过 1920px，总像素不超过 1920×1080；同一角色包内所有动作尺寸一致。
- 逐帧检查单张人脸、脸宽至少 160px、检测覆盖率至少 98%。不接受多人脸、连续三帧漏检；允许的极短漏检用相邻检测框修复，并在报告中列出帧号。覆盖率不是画面审美或姿态质量保证。
- 人物身份、服装、背景、光照、机位、头手是否完整、遮脸、口部自然闭合由人工确认；当前检测器不会自动判断这些条件。推荐每段 5–8 秒、动作始终接近正面、起止回到中性站姿。需人工检查往返播放的动作是否自然。
- 该流程严格校验，不自动转换编码或帧率。可用 FFmpeg 先导出符合要求的视频；导出不能修复换脸、模糊或遮挡问题。
- 原视频音频不参与生成；上传音频可通过 `/humanaudio` 驱动口型。LiveTalking 水印保留。

## 网页工作流

创建角色包，编辑动作条目（必须含 `neutral_explain`），逐个上传视频并勾选画面确认。任务在现有单线程生成队列中依次执行；验证和解析状态持久化。失败动作可重新上传，已完成或已发布动作禁止覆盖，需要更新时创建新角色 ID。

每个已完成动作可预览原视频或连接数字人并上传测试音频。全部动作完成后可“发布并重启”：发布窗口确认后设置中性讲解为默认 Avatar，当前连接断开，容器约 2 秒后由 Docker 自动重启。其他动作按需加载，不把所有片段常驻内存。页面会自动重连后台状态，不会自动恢复 WebRTC。

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
