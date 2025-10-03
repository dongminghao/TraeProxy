from flask import Flask, request, Response, jsonify
import requests
import json
import uuid
from typing import Dict, Optional

# -------------------------- 核心配置（可根据需求调整）--------------------------
# 1. NVIDIA 大模型 API 配置（参考官方文档，需替换为你的 NVIDIA API Key）
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"  # 从 NVIDIA NGC 平台获取

# 2. 代理服务配置（Trae IDE 连接时需使用以下信息）
PROXY_API_KEY = "sk-nvidia-trae-proxy-87654321-ABCD-EFGH-IJKL-1234567890AB"  # 自定义API密钥（Trae中需填入）
PROXY_HOST = "0.0.0.0"  # 允许外部访问（本地部署填127.0.0.1）
PROXY_PORT = 443        # 代理服务端口

# 3. 模型参数映射表（key=代理模型ID，value=模型个性化参数）
#    注：Trae中添加模型时，"模型ID"需填写此处的key（如"nvidia/llama-3.3-nemotpython trae_nvidia_proxy.pyron"）
MODEL_CONFIGS: Dict[str, Dict] = {
    # 模型1：NVIDIA Llama 3.3 Nemotron Super
    "nvidia/llama-3.3-nemotron": {
        "nvidia_model_id": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 65536,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": True,
        "system_content": "/think"  # 该模型需额外添加system prompt
    },
    # 模型2：NVIDIA Nemotron Nano
    "nvidia/nemotron-nano": {
        "nvidia_model_id": "nvidia/nvidia-nemotron-nano-9b-v2",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2048,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": True,
        "system_content": "/think",
        "extra_body": {  # 模型特有额外参数
            "min_thinking_tokens": 1024,
            "max_thinking_tokens": 2048
        }
    },
    # 模型3：Qwen3 Next 80B
    "qwen/qwen3-next-80b-a3b-instruct": {
        "nvidia_model_id": "qwen/qwen3-next-80b-a3b-instruct",
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 4096,
        "stream": True
    },
    # 模型4：DeepSeek V3.1
    "deepseek-ai/deepseek-v3.1": {
        "nvidia_model_id": "deepseek-ai/deepseek-v3.1",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 8192,
        "stream": True,
        "extra_body": {
            "chat_template_kwargs": {"thinking": True}
        }
    },
    # 模型5：GPT-OSS 120B
    "openai/gpt-oss-120b": {
        "nvidia_model_id": "openai/gpt-oss-120b",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "stream": True
    },
    # 模型6：Qwen3 Coder 480B
    "qwen/qwen3-coder-480b-a35b-instruct": {
        "nvidia_model_id": "qwen/qwen3-coder-480b-a35b-instruct",
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 4096,
        "stream": True
    },
    # 模型7：deepseek-r1-0528
    "deepseek-ai/deepseek-r1-0528": {
        "nvidia_model_id": "deepseek-ai/deepseek-r1-0528",
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 4096,
        "stream": True
    },
    # 模型8：kimi-k2
    "moonshotai/kimi-k2-instruct-0905": {
        "nvidia_model_id": "moonshotai/kimi-k2-instruct-0905",
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": True
    }   
}

# -------------------------- 代理服务实现 --------------------------
app = Flask(__name__)

def validate_api_key() -> bool:
    """验证请求中的API密钥是否匹配代理配置"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    return auth_header.split("Bearer ")[1] == PROXY_API_KEY

def generate_response_id() -> str:
    """生成符合Trae要求的响应ID（UUID去掉横杠）"""
    return str(uuid.uuid4()).replace("-", "")

@app.route("/v1/models", methods=["GET"])
def get_models():
    """
    Trae IDE 必备接口：返回可接入的模型列表
    Trae会检查该接口返回的模型ID是否与用户填写的一致
    """
    # 构造Trae要求的模型列表格式（必须包含id、object、owned_by字段）
    models_list = [
        {
            "id": model_id,  # 代理模型ID（Trae中需填写此值）
            "object": "model",
            #"owned_by": "NVIDIA + Trae Proxy"  # 固定值，Trae会检查该字段
            "owned_by": model_id.split("/")[0]
        }
        for model_id in MODEL_CONFIGS.keys()
    ]
    return jsonify({
        "object": "list",
        "data": models_list
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    Trae IDE 核心接口：转发聊天请求至NVIDIA大模型
    处理流式响应，并适配Trae要求的格式
    """
    # 1. 验证API密钥
    if not validate_api_key():
        return jsonify({"error": "Invalid API Key"}), 401

    # 2. 解析Trae的请求参数
    request_data = request.json
    proxy_model_id = request_data.get("model")  # Trae传入的代理模型ID
    messages = request_data.get("messages", [])  # Trae传入的对话内容

    # 3. 检查模型ID是否存在于配置中
    if proxy_model_id not in MODEL_CONFIGS:
        return jsonify({"error": f"Model {proxy_model_id} not found"}), 404
    model_config = MODEL_CONFIGS[proxy_model_id]

    # 4. 构造发给NVIDIA的请求参数
    nvidia_payload = {
        "model": model_config["nvidia_model_id"],
        "messages": messages,
        "temperature": model_config["temperature"],
        "top_p": model_config["top_p"],
        "max_tokens": model_config["max_tokens"],
        "stream": model_config["stream"]
    }
    # 添加模型特有参数（如frequency_penalty、extra_body）
    if "frequency_penalty" in model_config:
        nvidia_payload["frequency_penalty"] = model_config["frequency_penalty"]
    if "presence_penalty" in model_config:
        nvidia_payload["presence_penalty"] = model_config["presence_penalty"]
    # 合并extra_body（如min_thinking_tokens、chat_template_kwargs）
    if "extra_body" in model_config:
        nvidia_payload.update(model_config["extra_body"])
    # 添加模型要求的system prompt（如/think）
    if "system_content" in model_config:
        nvidia_payload["messages"].insert(0, {
            "role": "system",
            "content": model_config["system_content"]
        })

    # 5. 发送请求至NVIDIA API（流式响应）
    nvidia_headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        nvidia_response = requests.post(
            url=f"{NVIDIA_BASE_URL}/chat/completions",
            json=nvidia_payload,
            headers=nvidia_headers,
            stream=True,
            timeout=300  # 大模型响应较慢，延长超时时间
        )
        nvidia_response.raise_for_status()  # 抛出HTTP错误（如401、404）
    except Exception as e:
        return jsonify({"error": f"NVIDIA API request failed: {str(e)}"}), 500

    # 6. 处理流式响应，适配Trae格式（SSE协议，用\n\n分割消息）
    response_id = generate_response_id()
    # 正确解析HTTP日期头部以获得时间戳
    date_header = nvidia_response.headers.get("date", "")
    if date_header:
        # HTTP日期格式: "Thu, 01 Jan 1970 00:00:00 GMT"
        # 使用email.utils.parsedate_to_datetime来解析
        from email.utils import parsedate_to_datetime
        try:
            created_time = int(parsedate_to_datetime(date_header).timestamp())
        except Exception:
            # 如果解析失败，使用当前时间
            import time
            created_time = int(time.time())
    else:
        # 如果没有日期头部，使用当前时间
        import time
        created_time = int(time.time())

    def stream_generator():
        # 首条响应：必须包含role=assistant和空content（Trae要求）
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": proxy_model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        # 中间响应：转发NVIDIA的流式内容（去掉role字段）
        finish_reason = None
        for line in nvidia_response.iter_lines(decode_unicode=True):
            if not line.startswith("data: "):
                continue
            nvidia_chunk = line[6:]  # 去掉"data: "前缀
            if nvidia_chunk == "[DONE]":
                finish_reason = "stop"
                break
            try:
                chunk_data = json.loads(nvidia_chunk)
                # 适配Trae格式：移除delta中的role，保留content
                if "choices" in chunk_data and chunk_data["choices"]:
                    chunk_data["id"] = response_id
                    chunk_data["model"] = proxy_model_id
                    chunk_data["choices"][0]["delta"].pop("role", None)  # Trae后续响应不需要role
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            except json.JSONDecodeError:
                continue

        # 末尾响应：必须包含空content、finish_reason=stop和usage（Trae要求）
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": proxy_model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": finish_reason or "stop"
                }
            ],
            "usage": {  # 若NVIDIA返回usage可替换，此处默认填充0
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        # 发送结束标志
        yield "data: [DONE]\n\n"

    # 返回流式响应（SSE格式）
    return Response(
        stream_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Trae-NVIDIA 代理服务启动信息：")
    print(f"1. 代理地址：http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"2. Trae 模型ID（任选其一）：{list(MODEL_CONFIGS.keys())}")
    print(f"3. Trae API密钥：{PROXY_API_KEY}")
    print(f"4. NVIDIA API密钥状态：{'已配置' if NVIDIA_API_KEY != '你的NVIDIA API Key' else '未配置（需替换）'}")
    print("=" * 60)
    app.run(host=PROXY_HOST, port=PROXY_PORT, debug=False,ssl_context=("certs/api.openai.com.crt", "certs/api.openai.com.key"))  # 生产环境关闭debug