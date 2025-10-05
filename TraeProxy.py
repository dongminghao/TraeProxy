from flask import Flask, request, Response, jsonify
import requests
import json
import uuid
import threading
import queue
import concurrent.futures
import atexit
import os
import time
from typing import Dict, Optional, Generator, Any, Tuple

# -------------------------- 核心配置（可根据需求调整）--------------------------
# 1. NVIDIA 大模型 API 配置（参考官方文档，需替换为你的 NVIDIA API Key）
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"  # 填入你自己的APIKEY,从 NVIDIA NGC 平台获取
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

# -------------------------- 双线程双缓冲配置 --------------------------
# 支持从环境变量动态配置
def get_env_int(name: str, default: int) -> int:
    """从环境变量获取整数值，如果不存在或无效则使用默认值"""
    try:
        value = os.environ.get(name)
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        logger.warning(f"无效的环境变量 {name}，使用默认值: {default}")
        return default

MAX_QUEUE_SIZE = get_env_int("TRAE_PROXY_MAX_QUEUE_SIZE", 100)  # 请求队列最大容量
MAX_WORKERS = get_env_int("TRAE_PROXY_MAX_WORKERS", 20)        # 处理NVIDIA请求的线程池最大数量
MAX_RETRY_COUNT = get_env_int("TRAE_PROXY_MAX_RETRY", 3)       # 请求失败最大重试次数
SHUTDOWN_TIMEOUT = get_env_int("TRAE_PROXY_SHUTDOWN_TIMEOUT", 30)  # 关闭超时时间(秒)
REQUEST_TIMEOUT = get_env_int("TRAE_PROXY_REQUEST_TIMEOUT", 300)  # 请求超时时间（秒）
STREAM_MAX_TIMEOUT = get_env_int("TRAE_PROXY_STREAM_MAX_TIMEOUT", 60) # 流式响应最大超时时间（秒）

# 移除配置日志记录以提高性能

# 请求队列
request_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# 线程安全的共享资源
response_queues_lock = threading.Lock()  # 保护response_queues字典的锁
response_events_lock = threading.Lock()  # 保护response_events字典的锁
response_queues: Dict[str, queue.Queue] = {}
response_events: Dict[str, threading.Event] = {}  # 用于通知响应生成器NVIDIA请求已完成

# 线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# 优雅关闭控制
shutdown_event = threading.Event()  # 用于通知所有线程关闭

# -------------------------- 代理服务实现 --------------------------
app = Flask(__name__)

def process_nvidia_request(request_data: Dict, response_queue: queue.Queue, done_event: threading.Event):
    """
    在独立线程中处理单个NVIDIA API请求，并将结果放入响应队列。
    优化锁使用和性能关键路径。
    """
    response_id = request_data["response_id"]
    proxy_model_id = request_data["proxy_model_id"]
    messages = request_data["messages"]
    
    # 添加请求状态监控
    request_start_time = time.time()
    response_queue.put({"status": "started", "timestamp": request_start_time, "model": proxy_model_id})
    
    # 使用get方法避免异常处理开销
    model_config = MODEL_CONFIGS.get(proxy_model_id)
    if not model_config:
        response_queue.put({"error": f"Model {proxy_model_id} configuration not found", "code": "model_not_found"})
        response_queue.put(None)
        done_event.set()
        return

    # 构造发给NVIDIA的请求参数 - 使用更高效的字典构建方式
    nvidia_payload = {
        "model": model_config["nvidia_model_id"],
        "messages": messages.copy(),  # 创建副本避免修改原始数据
        "temperature": model_config["temperature"],
        "top_p": model_config["top_p"],
        "max_tokens": model_config["max_tokens"],
        "stream": model_config["stream"]
    }
    
    # 使用字典推导式合并可选参数，减少条件判断
    optional_params = {k: model_config[k] for k in ["frequency_penalty", "presence_penalty"] if k in model_config}
    nvidia_payload.update(optional_params)
    
    if "extra_body" in model_config:
        nvidia_payload.update(model_config["extra_body"])
    if "system_content" in model_config:
        nvidia_payload["messages"].insert(0, {"role": "system", "content": model_config["system_content"]})

    # 发送请求至NVIDIA API，添加重试机制
    nvidia_headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    retry_count = 0
    while retry_count < MAX_RETRY_COUNT:
        # 快速检查关闭信号
        if shutdown_event.is_set():
            response_queue.put({"error": "Server shutting down", "code": "server_shutdown"})
            response_queue.put(None)
            done_event.set()
            return
            
        # 在请求处理过程中定期发送状态更新
        if time.time() - request_start_time > 30:  # 每30秒发送一次状态更新
            response_queue.put({"status": "processing", "elapsed": time.time() - request_start_time, "model": proxy_model_id})
            
        try:
            nvidia_response = requests.post(
                url=f"{NVIDIA_BASE_URL}/chat/completions",
                json=nvidia_payload,
                headers=nvidia_headers,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            
            # 检查HTTP错误 - 优化错误处理路径
            if nvidia_response.status_code >= 400:
                error_code = f"nvidia_api_error_{nvidia_response.status_code}"
                error_message = f"NVIDIA API returned error: {nvidia_response.status_code}"
                
                # 只在客户端错误时尝试解析JSON
                if nvidia_response.status_code < 500:
                    try:
                        error_json = nvidia_response.json()
                        if "error" in error_json:
                            error_message = f"NVIDIA API error: {error_json['error'].get('message', 'Unknown error')}"
                    except:
                        pass
                
                # 对更多错误类型进行重试，包括特定于deepseek模型的错误
                if ((nvidia_response.status_code >= 500 or 
                    nvidia_response.status_code == 429 or  # 请求过多
                    (nvidia_response.status_code >= 400 and "deepseek" in proxy_model_id.lower())) and  # 对deepseek模型的客户端错误也重试
                    retry_count < MAX_RETRY_COUNT - 1):
                    retry_count += 1
                    # 使用指数退避
                    backoff_time = min(2 ** retry_count, 10)  # 指数退避，最大10秒
                    time.sleep(backoff_time)
                    continue
                
                response_queue.put({"error": error_message, "code": error_code})
                response_queue.put(None)
                done_event.set()
                return
            
            # 处理流式响应 - 优化数据处理路径，添加流量控制
            line_count = 0
            for line in nvidia_response.iter_lines(decode_unicode=True):
                if shutdown_event.is_set():
                    break
                    
                if line and line.startswith("data: "):
                    # 实现流量控制：每处理10行数据，检查队列大小
                    line_count += 1
                    if line_count % 10 == 0:
                        # 检查队列大小，如果队列过大则暂停处理
                        try:
                            queue_size = response_queue.qsize()
                            if queue_size > 50:  # 如果队列中有超过50个项目
                                time.sleep(0.1)  # 暂停0.1秒，让消费者处理队列中的数据
                        except:
                            # 如果无法获取队列大小，继续处理
                            pass
                    
                    # 使用非阻塞方式放入队列，避免阻塞
                    try:
                        response_queue.put(line, block=False)
                    except queue.Full:
                        # 如果队列已满，等待一小段时间后重试
                        time.sleep(0.05)
                        try:
                            response_queue.put(line, block=False)
                        except queue.Full:
                            # 如果仍然失败，记录错误并继续
                            # 在实际应用中，这里可以添加日志记录
                            pass
            
            # 流结束后放入None作为标志
            response_queue.put(None)
            done_event.set()
            return  # 成功完成，直接返回
            
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count < MAX_RETRY_COUNT:
                # 使用指数退避
                backoff_time = min(2 ** retry_count, 10)  # 指数退避，最大10秒
                time.sleep(backoff_time)
                continue
            response_queue.put({"error": f"NVIDIA API request timed out after {REQUEST_TIMEOUT} seconds", "code": "nvidia_api_timeout"})
            response_queue.put(None)
            break
            
        except requests.exceptions.ConnectionError:
            retry_count += 1
            if retry_count < MAX_RETRY_COUNT:
                # 使用指数退避
                backoff_time = min(2 ** retry_count, 10)  # 指数退避，最大10秒
                time.sleep(backoff_time)
                continue
            response_queue.put({"error": "Connection error when connecting to NVIDIA API", "code": "nvidia_api_connection_error"})
            response_queue.put(None)
            break
            
        except requests.exceptions.RequestException as e:
            # 处理请求相关的异常
            retry_count += 1
            if retry_count < MAX_RETRY_COUNT:
                # 使用指数退避
                backoff_time = min(2 ** retry_count, 10)  # 指数退避，最大10秒
                time.sleep(backoff_time)
                continue
            response_queue.put({"error": f"Request error: {str(e)}", "code": "request_error"})
            response_queue.put(None)
            break
            
        except json.JSONDecodeError as e:
            # 处理JSON解析错误
            response_queue.put({"error": f"JSON decode error: {str(e)}", "code": "json_decode_error"})
            response_queue.put(None)
            break
            
        except Exception as e:
            # 处理其他异常，添加模型特定信息
            error_info = f"NVIDIA API request failed for model {proxy_model_id}: {str(e)}"
            response_queue.put({"error": error_info, "code": "nvidia_api_error"})
            response_queue.put(None)
            break
    
    # 确保事件被设置
    done_event.set()

def request_processor():
    """
    请求处理线程：从请求队列中获取任务，并提交到线程池执行。
    保留线程安全保护和优雅关闭支持，但移除日志记录以提高性能。
    """
    while not shutdown_event.is_set():
        try:
            # 使用超时获取，以便定期检查关闭信号
            request_data = request_queue.get(timeout=1.0)
            
            if request_data is None:  # 终止信号
                break
            
            response_id = request_data["response_id"]
            
            # 线程安全地添加队列和事件
            with response_queues_lock:
                response_queues[response_id] = queue.Queue()
            
            with response_events_lock:
                response_events[response_id] = threading.Event()
            
            # 提交任务到线程池
            executor.submit(
                process_nvidia_request, 
                request_data, 
                response_queues[response_id], 
                response_events[response_id]
            )
            
        except queue.Empty:
            # 超时，继续循环并检查关闭信号
            continue
        except Exception:
            # 继续处理其他请求
            continue

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
    # 检查服务是否正在关闭
    if shutdown_event.is_set():
        return jsonify({"error": "Service is shutting down", "code": "service_unavailable"}), 503
        
    # 1. 验证API密钥
    if not validate_api_key():
        return jsonify({"error": "Invalid API Key", "code": "invalid_api_key"}), 401

    try:
        # 2. 解析Trae的请求参数
        request_data = request.json
        if not request_data:
            return jsonify({"error": "Empty request body", "code": "invalid_request"}), 400
            
        proxy_model_id = request_data.get("model")
        if not proxy_model_id:
            return jsonify({"error": "Missing model parameter", "code": "invalid_request"}), 400
            
        messages = request_data.get("messages", [])
        if not messages:
            return jsonify({"error": "Empty messages array", "code": "invalid_request"}), 400

        # 3. 检查模型ID是否存在于配置中
        if proxy_model_id not in MODEL_CONFIGS:
            return jsonify({"error": f"Model {proxy_model_id} not found", "code": "model_not_found"}), 404

        # 4. 创建唯一响应ID，并将请求放入队列
        response_id = generate_response_id()
        
        # 5. 构造请求数据并放入队列
        queued_request = {
            "response_id": response_id,
            "proxy_model_id": proxy_model_id,
            "messages": messages,
            "timestamp": time.time()
        }
        
        try:
            # 使用超时，避免无限等待
            request_queue.put(queued_request, block=True, timeout=5)
        except queue.Full:
            return jsonify({"error": "Request queue is full, server is overloaded", "code": "server_overloaded"}), 503

        # 6. 返回流式响应
        return Response(
            stream_generator(response_id, proxy_model_id),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}", "code": "internal_error"}), 500

def stream_generator(response_id: str, proxy_model_id: str) -> Generator[str, None, None]:
    """
    从指定响应队列中获取数据，并适配Trae格式生成流式响应。
    """
    # 检查服务是否正在关闭
    if shutdown_event.is_set():
        error_chunk = {"id": response_id, "choices": [{"delta": {"content": "\n\nError: Service is shutting down."}}]}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
        
    created_time = int(uuid.uuid4().int / 1e6) # 模拟创建时间

    # 首条响应
    first_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": proxy_model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    # 中间响应
    finish_reason = None
    start_time = time.time()
    
    # 线程安全地获取响应队列
    with response_queues_lock:
        response_queue = response_queues.get(response_id)
    
    if not response_queue:
        # 队列可能还未创建，稍作等待，特别是对于deepseek模型
        with response_events_lock:
            if response_id in response_events:
                # 增加等待时间，特别是对于deepseek模型
                wait_timeout = 10 if "deepseek" in proxy_model_id.lower() else 5
                response_events[response_id].wait(timeout=wait_timeout)
        
        # 再次尝试获取队列（线程安全）
        with response_queues_lock:
            response_queue = response_queues.get(response_id)
            
        if not response_queue:
            # 如果还未创建，说明请求处理失败
            error_chunk = {"id": response_id, "choices": [{"delta": {"content": "\n\nError: Request processing failed to start."}}]}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

    while not shutdown_event.is_set():
        try:
            # 使用更长的超时时间获取，避免大流量输出时频繁超时
            try:
                line = response_queue.get(timeout=3.0)  # 增加到3秒超时
            except queue.Empty:
                # 检查总时间是否超过最大超时时间（防止无限等待）
                # 为deepseek模型使用更长的超时时间
                model_timeout = STREAM_MAX_TIMEOUT
                if "deepseek" in proxy_model_id.lower():
                    model_timeout = STREAM_MAX_TIMEOUT * 1.5  # deepseek模型使用1.5倍超时时间
                
                if time.time() - start_time > model_timeout:
                    error_chunk = {"id": response_id, "choices": [{"delta": {"content": f"\n\nError: Maximum streaming time ({model_timeout}s) exceeded."}}]}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    finish_reason = "timeout"
                    break
                
                # 添加队列状态检查，区分临时空队列和真正超时
                empty_count = getattr(stream_generator, 'empty_count', 0) + 1
                setattr(stream_generator, 'empty_count', empty_count)
                
                # 如果连续5次队列为空，发送一个状态更新
                if empty_count % 5 == 0:
                    status_chunk = {"id": response_id, "choices": [{"delta": {"content": f"\n\nStatus: Processing large response, please wait..."}}]}
                    yield f"data: {json.dumps(status_chunk)}\n\n"
                
                continue
                
            if line is None:
                break
            
            if isinstance(line, dict) and "error" in line:
                # 处理错误信息
                error_content = line["error"]
                error_chunk = {"id": response_id, "choices": [{"delta": {"content": f"\n\nError: {error_content}"}}]}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                finish_reason = "error"
                break
            
            # 处理状态更新信息
            if isinstance(line, dict) and "status" in line:
                # 状态更新信息，不传递给客户端，仅用于内部监控
                continue

            if not line.startswith("data: "):
                continue
            
            nvidia_chunk_str = line[6:]
            if nvidia_chunk_str == "[DONE]":
                finish_reason = "stop"
                continue
                
            try:
                chunk_data = json.loads(nvidia_chunk_str)
                if "choices" in chunk_data and chunk_data["choices"]:
                    chunk_data["id"] = response_id
                    chunk_data["model"] = proxy_model_id
                    chunk_data["choices"][0]["delta"].pop("role", None)
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            except json.JSONDecodeError as json_error:
                # 增强JSON解析错误处理，记录错误内容但不中断流
                # 在实际应用中，这里可以添加日志记录
                continue
            except Exception as chunk_error:
                # 处理其他数据块处理错误
                # 在实际应用中，这里可以添加日志记录
                continue
        except Exception as e:
            # 增强异常处理，提供更详细的错误信息
            error_type = type(e).__name__
            error_detail = str(e)
            
            # 根据错误类型提供更具体的错误信息
            if "queue" in error_type.lower() or "Queue" in error_type:
                error_message = f"\n\nError: Queue operation failed ({error_type}): {error_detail}"
            elif "timeout" in error_type.lower() or "Timeout" in error_type:
                error_message = f"\n\nError: Operation timed out ({error_type}): {error_detail}"
            elif "connection" in error_type.lower() or "Connection" in error_type:
                error_message = f"\n\nError: Connection issue ({error_type}): {error_detail}"
            else:
                error_message = f"\n\nError: Stream processing failed ({error_type}): {error_detail}"
            
            error_chunk = {"id": response_id, "choices": [{"delta": {"content": error_message}}]}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            finish_reason = "error"
            break

    # 末尾响应
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": proxy_model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    # 清理资源（线程安全）
    with response_queues_lock:
        if response_id in response_queues:
            del response_queues[response_id]
    with response_events_lock:
        if response_id in response_events:
            del response_events[response_id]


def shutdown_threads():
    """优雅地关闭线程池和请求处理线程"""
    # 设置关闭事件，通知所有线程准备关闭
    shutdown_event.set()
    request_queue.put(None)  # 发送终止信号
    # 等待线程池完成当前任务，最多等待SHUTDOWN_TIMEOUT秒
    executor.shutdown(wait=True, timeout=SHUTDOWN_TIMEOUT)
    # 清理所有剩余的响应队列和事件
    with response_queues_lock:
        response_queues.clear()
    with response_events_lock:
        response_events.clear()

if __name__ == "__main__":
    # 启动请求处理线程
    processor_thread = threading.Thread(target=request_processor)
    processor_thread.daemon = True
    processor_thread.start()

    # 注册退出时关闭线程的函数
    atexit.register(shutdown_threads)

    print("=" * 60)
    print("Trae-NVIDIA 代理服务启动信息：")
    print(f"1. 代理地址：http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"2. Trae 模型ID（任选其一）：{list(MODEL_CONFIGS.keys())}")
    print(f"3. Trae API密钥：{PROXY_API_KEY}")
    print(f"4. NVIDIA API密钥状态：{'已配置' if NVIDIA_API_KEY != '你的NVIDIA API Key' else '未配置（需替换）'}")
    print("=" * 60)
    app.run(host=PROXY_HOST, port=PROXY_PORT, debug=False,ssl_context=("certs/api.openai.com.crt", "certs/api.openai.com.key"))  # 生产环境关闭debug