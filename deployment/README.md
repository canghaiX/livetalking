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

浏览器访问 `http://222.30.145.22:8020/index.html`，点击开始连接。默认 EdgeTTS 提供文本播报，需要能够访问微软语音服务；音频上传驱动可直接使用。LLM 对话需用户自己的 API Key，例如通过容器环境变量 `DASHSCOPE_API_KEY` 配置；本部署没有预置外部服务密钥。

本次验证：容器健康检查通过，CUDA 只看到 GPU 0（RTX A6000），`pip check` 通过。WebRTC 实测已收到音视频，并使用上传的 16kHz 音频成功触发口型推理；测试结果见 `deployment/webrtc-smoke-test.log`。默认 EdgeTTS 7.2.8 实际测试被微软服务返回 HTTP 403，文本合成尚不可用，日志见 `deployment/tts-check.log`；大屏演示可先上传预录音频，或配置用户可用的 TTS 服务。

镜像安装的 Python 包版本可从 `/opt/livetalking/requirements.installed.txt` 获取。服务器外部访问需要 TCP 8020 和 WebRTC 协商的 UDP 端口可达。

角色包管理页面为 `/avatar-package.html`，操作、素材要求及接口见 [角色包说明](../docs/avatar_packages.md)。各动作分别生成 Avatar，预览时重新建立 WebRTC 连接。发布会保存 manifest 和默认动作，再让主进程退出，由 Docker `unless-stopped` 策略重启；不会挂载 Docker socket。S3FD 检测权重必须预先准备在 `models/s3fd.pth`，避免生成时临时访问外网。

更新镜像后需要重建容器以应用新镜像和挂载（持久化目录不会被删除）：

```bash
docker stop livetalking
docker rm livetalking
bash deployment/start.sh
```

角色包验证包括 14 项自动测试、五动作生成及 WebRTC 上传音频集成测试、发布重启恢复，以及桌面和手机端网页上传、失败重试与动作连接测试。集成测试使用默认 Avatar 的短视频作为五个动作的测试素材；实际铁路讲解形象仍需上传用户准备的素材。模型权重、生成数据、凭据与运行日志不提交到 Git。
