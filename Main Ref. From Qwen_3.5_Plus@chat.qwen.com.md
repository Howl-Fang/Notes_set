***Question:
https://github.com/zai-org/Open-AutoGLM 
请分析这个项目。提供如何部署的信息

# 🔍 Open-AutoGLM 项目分析与部署指南

## 📋 项目概述

**Open-AutoGLM** 是由智谱AI开源的**手机端智能助理框架**，基于 AutoGLM 构建，能够以多模态方式理解手机屏幕内容，并通过自动化操作帮助用户完成任务。

### 核心工作原理
```
用户自然语言指令 → 截图 → 视觉模型理解界面 → 输出操作坐标 → ADB执行 → 循环迭代
```

### 架构组成
| 组件 | 说明 | 运行位置 |
|------|------|----------|
| **Agent 代码** | 调用模型、解析动作、控制手机 | 用户电脑/服务器 |
| **视觉模型服务** | 屏幕理解与决策生成 | 远程API 或 本地部署 |

### 支持设备
- ✅ Android 7.0+（通过 ADB）
- ✅ HarmonyOS NEXT+（通过 HDC）
- ✅ iOS（需单独配置 WebDriverAgent）

---

## 🚀 部署完整流程

### 🔧 前置准备

#### 硬件要求
- Android 7.0+ 或 HarmonyOS 设备
- 支持**数据传输**的 USB 数据线（非仅充电线）
- 电脑（Windows/Mac/Linux）

#### 手机端配置
```bash
# 1. 开启开发者模式
设置 → 关于手机 → 连续点击"版本号"7次

# 2. 开启 USB 调试
设置 → 开发者选项 → USB 调试 ✓

# 3. （部分机型）开启安全调试
设置 → 开发者选项 → USB 调试(安全设置) ✓

# 4. Android 设备需安装 ADB Keyboard
# 下载地址: https://github.com/senzhk/ADBKeyBoard
# 安装后在 设置→输入法 中启用
```

#### 电脑端安装 ADB/HDC
```bash
# MacOS 安装 ADB
brew install android-platform-tools

# 或手动下载: https://developer.android.com/tools/releases/platform-tools

# 配置环境变量 (MacOS/Linux)
export PATH=${PATH}:~/Downloads/platform-tools

# 验证安装
adb version  # 应输出版本信息
```

---

### 📦 安装 Agent 代码

```bash
# 1. 克隆仓库
git clone https://github.com/zai-org/Open-AutoGLM.git
cd Open-AutoGLM

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .
```

---

### 🤖 配置模型服务（二选一）

#### ✅ 选项 A：使用第三方 API（推荐新手）

| 服务商 | base-url | model | 说明 |
|--------|----------|-------|------|
| **智谱 BigModel** | `https://open.bigmodel.cn/api/paas/v4` | `autoglm-phone` | 中文优化 |
| **z.ai (国际)** | `https://api.z.ai/api/paas/v4` | `autoglm-phone-multilingual` | 多语言支持 |
| **ModelScope** | `https://api-inference.modelscope.cn/v1` | `ZhipuAI/AutoGLM-Phone-9B` | 魔搭社区 |
| **Novita AI** | `https://api.novita.ai/openai` | `zai-org/autoglm-phone-9b-multilingual` | 国际服务 |

**使用示例：**
```bash
# 使用智谱 BigModel
python main.py \
  --base-url https://open.bigmodel.cn/api/paas/v4 \
  --model "autoglm-phone" \
  --apikey "your-api-key" \
  "打开美团搜索附近的火锅店"
```

#### ⚙️ 选项 B：本地部署模型（高配置要求）

**硬件要求**：NVIDIA GPU，建议 24GB+ 显存

```bash
# 1. 安装推理引擎
pip install vllm  # 或 sglang

# 2. 启动 vLLM 服务（严格使用以下参数）
python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name autoglm-phone-9b \
  --allowed-local-media-path / \
  --mm-encoder-tp-mode data \
  --mm_processor_cache_type shm \
  --mm_processor_kwargs "{\"max_pixels\":5000000}" \
  --max-model-len 25480 \
  --chat-template-content-format string \
  --limit-mm-per-prompt "{\"image\":10}" \
  --model zai-org/AutoGLM-Phone-9B \
  --port 8000
```

> 📦 模型会自动下载，约 20GB。服务地址：`http://localhost:8000/v1`

---

### 🔍 验证部署

```bash
# 检查模型服务
python scripts/check_deployment_cn.py \
  --base-url http://localhost:8000/v1 \
  --model autoglm-phone-9b

# 测试任务执行
python main.py \
  --base-url {MODEL_URL} \
  --model "autoglm-phone-9b" \
  "打开微信，对文件传输助手发送消息：部署成功"
```

✅ **成功标志**：手机自动打开微信 → 搜索联系人 → 发送消息

---

## 🎮 使用方式

### 命令行模式
```bash
# 交互模式（可连续对话）
python main.py --base-url {URL} --model {MODEL_NAME}

# 单次任务
python main.py --base-url {URL} "打开淘宝搜索无线耳机"

# 鸿蒙设备
python main.py --device-type hdc --base-url {URL} "任务描述"

# 查看支持的应用
python main.py --list-apps
```

### Python API 调用
```python
from phone_agent import PhoneAgent
from phone_agent.model import ModelConfig

model_config = ModelConfig(
    base_url="http://localhost:8000/v1",
    model_name="autoglm-phone-9b",
)

agent = PhoneAgent(model_config=model_config)
result = agent.run("打开淘宝搜索无线耳机")
print(result)
```

---

## 🔧 常用配置

### 环境变量
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PHONE_AGENT_BASE_URL` | 模型 API 地址 | `http://localhost:8000/v1` |
| `PHONE_AGENT_MODEL` | 模型名称 | `autoglm-phone-9b` |
| `PHONE_AGENT_API_KEY` | API 认证密钥 | `EMPTY` |
| `PHONE_AGENT_MAX_STEPS` | 任务最大步数 | `100` |
| `PHONE_AGENT_DEVICE_TYPE` | 设备类型 `adb`/`hdc` | `adb` |

### 支持的操作类型
| 操作 | 描述 |
|------|------|
| `Launch` | 启动应用 |
| `Tap` | 点击坐标 |
| `Type` | 输入文本 |
| `Swipe` | 滑动屏幕 |
| `Back`/`Home` | 返回/主页 |
| `Take_over` | 人工接管（登录/验证码） |

---

## ⚠️ 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `adb devices` 无输出 | USB调试未开/线材问题 | 检查开发者选项，换数据线 |
| 显示 `unauthorized` | 手机未授权调试 | 手机弹窗点"允许" |
| 能打开App但无法点击 | 缺少安全调试权限 | 开启"USB调试(安全设置)" |
| 中文输入乱码 | ADB Keyboard 未启用 | 系统设置中启用输入法 |
| 截图黑屏 | 敏感页面(支付/银行) | 正常，系统会自动请求人工接管 |
| Windows 编码错误 | GBK 编码冲突 | 运行前加 `PYTHONIOENCODING=utf-8` |

---

## 🌐 远程调试（WiFi 连接）

```bash
# 手机端开启无线调试后，电脑端连接：
adb connect 192.168.1.100:5555  # 替换为手机实际IP

# 验证连接
adb devices  # 应显示: 192.168.1.100:5555    device

# 指定设备执行任务
python main.py --device-id 192.168.1.100:5555 \
  --base-url {URL} --model {MODEL} "任务描述"
```

---

## 📚 资源链接

- 🔗 项目主页：https://github.com/zai-org/Open-AutoGLM
- 📥 模型下载：[HuggingFace](https://huggingface.co/zai-org) | [ModelScope](https://modelscope.cn/models/ZhipuAI)
- 📖 官方文档：README.md / README_en.md
- 💬 社区：微信社群 / Discord

> ⚠️ **免责声明**：本项目仅供研究学习使用，严禁用于非法获取信息、干扰系统或任何违法活动。

如需更详细的某一步骤说明，欢迎继续提问！🎯