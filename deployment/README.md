# workspace 部署

项目仓库为 https://github.com/canghaiX/livetalking ，原始代码来自 https://github.com/lipku/livetalking（保留为 `upstream`），部署目录为 `/home/ym/workspace/livetalking`。

镜像 `livetalking:latest` 使用本机已存在的 `livetalking-main:local` 作为依赖基础（Python 3.11 / PyTorch 2.6 / CUDA 12.4），安装仓库 requirements，并将本次拉取的代码、Wav2Lip 模型和默认数字人打包到 `/opt/livetalking`。原仓库 Dockerfile 保留；本部署使用 `Dockerfile.workspace`。模型和 `wav2lip256_avatar1` 从服务器已有的 `/data/share/LiveTalking-main` 复制而来。

容器名为 `livetalking`，只开放宿主 GPU 0，使用 host 网络监听 8020，以支持 WebRTC UDP。配置文件 `deployment/config.yaml` 挂载到容器；`data/record`、`data/avatar_packages` 和 `data/avatars` 挂载到工作区持久化。配置修改后运行 `docker restart livetalking`。

```bash
cd /home/ym/workspace/livetalking
docker build -f Dockerfile.workspace --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t livetalking:latest .
bash deployment/start.sh
docker logs -f livetalking
docker stop livetalking
docker start livetalking
```

浏览器访问 `http://222.30.145.22:8020/index.html`，点击开始连接。默认 TTS 配置为阿里云百炼 Qwen 实时合成，固定使用 Serena 普通话女声；在 `deployment/secrets.env` 设置 `DASHSCOPE_API_KEY` 并重建容器后启用。阿里云临时超时会以 Serena 重建会话并退避重试，不会切换 EdgeTTS、Piper 或其他声线；用户打断后停止重试。密钥留在服务器本地，不要提交到 Git。

服务器网络代理由 `ym` 用户的 Mihomo 服务独立维护，配置文件位于 `/home/ym/.config/mihomo/config.yaml`，本地监听 `127.0.0.1:7890`，并通过 `systemd --user` 和 linger 开机启动。LiveTalking 容器固定使用该服务器代理，不依赖 VS Code、Codex 或本地电脑的 SSH 反向隧道。检查命令：`systemctl --user status mihomo.service`、`docker exec livetalking ...`。如果所有上游节点失效，需要更新该配置中的节点，不能改用 HTTP 80 代替 HTTPS 443。

TTS 会按语境转换数字读音，例如 `1950年` 读作“ 一九五零年”，`12306` 读作“ 一二三零六”；普通数量、日期、小数和百分数转换为中文数词。网页字幕和对话历史保留原始文本。音色与计费以[百炼官方说明](https://help.aliyun.com/zh/model-studio/qwen-tts-voice-list)为准。

首页为左侧数字人、右侧对话框。发送文字始终调用 DeepSeek，新消息或开始语音输入会打断上一轮回答；字幕按实际播报段显示原文。单视频和角色包管理页面保留，旧 `/human` 的 echo 接口仍兼容。

按住说话使用服务端阿里云 Paraformer 实时 ASR。浏览器仅发送 16kHz 单声道 PCM16 音频，不接触密钥。在服务器本地 `deployment/secrets.env` 中配置：

```dotenv
DASHSCOPE_API_KEY=<已有百炼密钥，TTS与ASR共用>
DASHSCOPE_WORKSPACE_ID=<北京地域业务空间ID，不是API Key的数字ID>
DASHSCOPE_ASR_MODEL=paraformer-realtime-v2
DASHSCOPE_ASR_MAX_SENTENCE_SILENCE=800
```

默认地址为 `wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference`，必要时通过 `DASHSCOPE_ASR_URL` 覆盖。环境变量修改后必须重建容器（仅 `docker restart` 不会重新读取 env-file）。`GET /api/asr/aliyun/status` 可检查缺少的配置，不返回密钥；缺少业务空间、密钥、权限或余额时页面明确报错，文字对话仍可使用。

麦克风必须在 HTTPS 或 localhost 安全上下文下使用。直接打开 HTTP IP 地址一般只能进行文字对话。测试时可在本地电脑执行 `ssh -L 8020:127.0.0.1:8020 ym@222.30.145.22`，再访问 `http://localhost:8020/index.html`；这只解决页面和麦克风安全上下文，WebRTC 的媒体连接仍需可达。正式大屏建议使用 HTTPS 反向代理并支持 WebSocket/SSE，保留 WebRTC UDP 连通。协议参考：[Paraformer WebSocket API](https://help.aliyun.com/zh/model-studio/websocket-for-paraformer-real-time-service)。

自动动作由 `deployment/avatar_motion.json` 配置并挂载：01作为待机和讲解底态，02、05、06暂停使用并保留文件；03只在明确介绍或引导时出现，使用统一构图、背景和光影后的 `neutral_explain_uniform_v3` 快速抬手版本；07使用 `gentle_introduce_07_uniform_v3` 完整正序播放温和介绍手势，04使用 `closing_thanks_uniform_v3` 在明确结束语音频完成后点头致意。新版本缺失时分别按 `*_facefix_v2`、旧背景修正版和原动作回退。03和07共享每轮最多三次、媒体时间间隔至少8秒的限制；两次动作之间至少有1秒01讲解。短暂停顿容忍0.6秒，切换使用6帧过渡。冷却中的手势只保留最新一个，到后续段落边界再检查；播报结束或打断时清除。不会恢复05/06。首页Avatar字段留空使用自动调度，显式选动作保留独立原视频预览。详情见[表情和手势播放](../docs/avatar_packages.md#表情和手势播放)。

本次验证：容器健康检查通过，CUDA 只看到 GPU 0（RTX A6000），`pip check` 通过。WebRTC 实测已收到音视频，并使用上传的 16kHz 音频成功触发口型推理；测试结果见 `deployment/webrtc-smoke-test.log`。生产 TTS 只允许阿里云 Qwen Serena，失败时不会静默切换为其他声线。

镜像安装的 Python 包版本可从 `/opt/livetalking/requirements.installed.txt` 获取。服务器外部访问需要 TCP 8020 和 WebRTC 协商的 UDP 端口可达。

角色包管理页面为 `/avatar-package.html`，操作、素材要求及接口见 [角色包说明](../docs/avatar_packages.md)。各动作分别生成 Avatar，预览时重新建立 WebRTC 连接。发布会保存 manifest 和默认动作，再让主进程退出，由 Docker `unless-stopped` 策略重启；不会挂载 Docker socket。S3FD 检测权重必须预先准备在 `models/s3fd.pth`，避免生成时临时访问外网。

更新镜像后需要重建容器以应用新镜像和挂载（持久化目录不会被删除）：

```bash
docker stop livetalking
docker rm livetalking
bash deployment/start.sh
```

角色包验证包括 14 项自动测试、五动作生成及 WebRTC 上传音频集成测试、发布重启恢复，以及桌面和手机端网页上传、失败重试与动作连接测试。集成测试使用默认 Avatar 的短视频作为五个动作的测试素材；实际铁路讲解形象仍需上传用户准备的素材。模型权重、生成数据、凭据与运行日志不提交到 Git。
### 播报与动作

Qwen TTS 固定使用 Serena。文本段由生成队列预取后再播放，网络短暂抖动时继续消费已缓存音频；不会回退到 EdgeTTS、Piper 或其他声线。长回答在自然段边界按约 35% 概率播放一次 `nod` 点头动作，语音不中断，动作完成后回到 01 微笑待机；明确 `closing` 仍只在最后一段音频完成后播放一次 04 致意。
