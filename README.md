# Trae IDE Proxy - 项目功能说明

## 项目概述

Trae IDE Proxy 是一个专为 Trae IDE 设计的代理服务，用于在 Trae IDE 和 NVIDIA 开源大模型及本地大模型之间建立连接，实现请求转发和响应处理。该项目允许用户在 Trae IDE 中无缝使用 NVIDIA 提供的各种高性能大语言模型，扩展了 Trae IDE 的 AI 辅助能力。

## 核心功能

### 1. 模型代理服务
- **多模型支持**：支持多种 NVIDIA 开源大模型，包括但不限于：
  - NVIDIA Llama 3.3 Nemotron Super
  - NVIDIA Nemotron Nano
  - Qwen3 Next 80B
  - DeepSeek V3.1
  - GPT-OSS 120B
  - Qwen3 Coder 480B
  - DeepSeek-R1-0528
  - Kimi-K2

- **模型参数自适应**：根据不同模型自动设置合适的参数（如温度、top_p、最大令牌数等），确保每个模型都能发挥最佳性能。

- **模型特有参数支持**：为特定模型提供额外的参数配置，如 thinking tokens、chat_template_kwargs 等。

### 2. API 接口兼容
- **OpenAI API 格式兼容**：完全兼容 OpenAI API 格式，便于与各种客户端集成。
- **标准端点实现**：
  - `POST /v1/chat/completions` - 聊天完成接口
  - `GET /v1/models` - 模型列表接口
  - `GET /` - 健康检查接口

### 3. 流式响应处理
- **实时响应**：支持流式响应，提供实时的文本生成体验。
- **响应格式适配**：将 NVIDIA API 的响应格式适配为 Trae IDE 所需的格式，包括：
  - 正确处理响应 ID 和时间戳
  - 适配消息块格式
  - 处理完成原因和使用统计信息

### 4. 安全与认证
- **API 密钥验证**：实现自定义 API 密钥验证机制，确保只有授权用户可以访问服务。
- **HTTPS 支持**：支持 HTTPS 连接，保障数据传输安全。

### 5. 配置灵活性
- **可配置参数**：提供丰富的配置选项，包括：
  - 代理服务地址和端口
  - NVIDIA API 密钥
  - 自定义代理 API 密钥
  - 各模型的个性化参数

- **易于扩展**：模块化设计，便于添加新的模型支持和功能扩展。

## 使用场景

### 1. Trae IDE 集成
- 在 Trae IDE 中添加自定义模型，通过代理服务访问 NVIDIA 大模型。
- 享受 Trae IDE 的开发环境优势，同时使用高性能的大语言模型。

### 2. 开发辅助
- 代码生成与补全
- 技术问题解答
- 文档生成与优化
- 代码审查与建议

### 3. 内容创作
- 文本生成与编辑
- 创意写作辅助
- 内容总结与提炼

## 技术特点

### 1. 高性能
- 优化的请求转发机制，最小化延迟
- 流式响应处理，提供实时反馈
- 高效的错误处理和恢复机制

### 2. 可靠性
- 完善的错误处理，包括网络错误、API 错误等
- 超时控制，防止长时间等待
- 日志记录，便于问题排查

### 3. 易用性
- 简单的安装和配置流程
- 详细的文档和使用示例
- 清晰的错误提示和解决方案

## 项目价值

1. **扩展 Trae IDE 能力**：为 Trae IDE 提供访问 NVIDIA 高性能大模型的通道，增强其 AI 辅助能力。
2. **降低使用门槛**：通过代理服务简化了 NVIDIA API 的使用，用户无需直接处理复杂的 API 调用。
3. **提高开发效率**：开发者可以在熟悉的环境中直接使用大语言模型，提高开发效率和代码质量。
4. **促进技术创新**：通过集成先进的大语言模型，促进 AI 辅助开发技术的应用和创新。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

```bash
python TraeProxy.py
```

默认运行在端口8080上。

## 在Trae IDE中配置自定义模型

在Trae IDE中添加自定义模型时，需要提供以下参数：

1. 服务商：NVIDIA
2. 模型：选择具体的模型名称（如 qwen/qwen3-coder-480b-a35b-instruct）
3. 模型ID：与模型名称相同
4. API密钥：您的NVIDIA API密钥

## API端点

- `POST /v1/chat/completions` - 聊天完成
- `GET /v1/models` - 列出可用模型
- `GET /` - 健康检查

## 使用示例

### 聊天完成请求

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_NVIDIA_API_KEY" \
  -d '{
    "model": "qwen/qwen3-coder-480b-a35b-instruct",
    "messages": [{"role": "user", "content": "Write a Python function to calculate Fibonacci numbers"}],
    "stream": true
  }'
```

### 列出模型

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer YOUR_NVIDIA_API_KEY"
```

## 配置前置条件

根据文档可知，Trae 接入本地 / 代理模型需满足域名劫持 + 证书信任两个前置条件，这是最易遗漏且直接导致请求失败的原因：

1. 未配置 hosts 文件，Trae 请求未指向代理服务

文档明确提到，需通过 hosts 文件将 api.openai.com 指向代理服务所在 IP（🔶1-20、🔶1-292）。若未配置，Trae 会默认请求真实的 OpenAI 服务器，而非你的本地代理，直接导致服务失败。

操作步骤：

- 找到系统 hosts 文件（Windows 路径：C:\Windows\System32\drivers\etc\hosts；Mac/Linux 路径：/etc/hosts）。
- 用管理员权限打开，添加一行：`127.0.0.1 api.openai.com`（若代理在其他设备，替换 127.0.0.1 为代理设备的 IP，如 192.168.0.3）。
- 保存后，在命令行执行 `ping api.openai.com`，确认返回 IP 为你配置的代理 IP（而非真实 OpenAI IP）。

2. 未安装并信任自签名证书，Trae 拒绝 HTTPS 连接

文档指出，Trae 请求 api.openai.com 时默认使用 HTTPS，需生成并信任对应域名的自签名证书（🔶1-186-190、🔶1-207-217）。你的代理服务当前是 HTTP（运行在 8080 端口），若未配置 HTTPS 转发 + 证书信任，Trae 会因 “证书不被信任” 或 “协议不匹配” 拒绝连接。

操作步骤（参考文档 “生成自签名证书” 和 “操作系统信任新证书” 章节）：

生成证书：在代理服务所在目录执行类似文档的脚本（🔶1-188），生成 ca.crt 和 api.openai.com.crt/key。

```bash
# 示例（可参考文档 vproxy 工程的 misc/ca 目录脚本）
mkdir -p certs && cd certs
# 生成 CA 根证书
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -days 3650 -out ca.crt -subj "/C=CN/CN=vproxy-ca"
# 生成 api.openai.com 域名证书
openssl genrsa -out api.openai.com.key 2048
openssl req -new -key api.openai.com.key -out api.openai.com.csr -subj "/C=CN/CN=api.openai.com"
openssl x509 -req -in api.openai.com.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out api.openai.com.crt -days 3650
```

注意：上述证书生成步骤仅供参考，具体操作请根据您的环境调整。

## 未来发展方向

1. **支持更多模型**：扩展支持更多的大语言模型，包括本地模型和其他云服务提供商的模型。

2. **增强功能**：添加更多高级功能，如对话历史管理、上下文控制、个性化设置等。

3. **性能优化**：进一步优化性能，减少延迟，提高吞吐量。

4. **安全增强**：加强安全措施，如请求限流、内容过滤等。

5. **监控与分析**：添加使用统计和性能监控功能，帮助用户了解模型使用情况。