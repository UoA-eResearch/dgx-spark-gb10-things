# Note, this works, does not use RDMA
import os
import sys

# --- STEP 1: SYSTEM & NETWORK CONFIGURATION ---
head_node_ip = "192.168.100.1"
target_interface = "enp1s0f0np0" 

# 1. COMPILER FIX (The new fix)
# Prevents "Exit status 137" by limiting JIT compilation to 1 thread
os.environ["MAX_JOBS"] = "1"
os.environ["FLASHINFER_MAX_JOBS"] = "1"
# Optional: Store compiled kernels so you don't have to wait next time
os.environ["TORCH_EXTENSIONS_DIR"] = "/root/.cache/torch_extensions"

# 2. Network - Force TCP/Sockets
os.environ["NCCL_SOCKET_IFNAME"] = target_interface
os.environ["GLOO_SOCKET_IFNAME"] = target_interface
os.environ["TP_SOCKET_IFNAME"] = target_interface
os.environ["NCCL_IB_DISABLE"] = "1"  # Force TCP
os.environ["NCCL_P2P_DISABLE"] = "1" # Disable P2P

# 3. Proxy
no_proxy_str = f"localhost,127.0.0.1,0.0.0.0,::1,{head_node_ip},192.168.100.2"
os.environ["no_proxy"] = no_proxy_str
os.environ["NO_PROXY"] = no_proxy_str
os.environ["RAY_grpc_enable_http_proxy"] = "0"

import ray
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# --- STEP 2: CONNECT ---
ray_addr = f"{head_node_ip}:6379"
print(f"🔗 Connecting to Ray at {ray_addr}...")
ray.init(address=ray_addr, ignore_reinit_error=True, log_to_driver=False, _node_ip_address=head_node_ip)

# --- STEP 3: INITIALIZE MODEL ---
print(f"🧠 Loading {model_name}...")
print("⏳ NOTE: First run may pause for minutes to compile Blackwell FP8 kernels.")
print("   Please be patient if it hangs on 'User: ' input...")

llm = LLM(
    model=model_name,
    tensor_parallel_size=2,
    trust_remote_code=True,
    # Try FP8, but if this still fails, change to "bfloat16"
    #quantization="fp8",
    dtype="bfloat16",
    gpu_memory_utilization=0.70,
    enforce_eager=True,
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)

# --- STEP 4: INTERACTIVE LOOP ---
print("\n" + "="*50)
print(f"🤖 Connected to {model_name} on 2x DGX Spark")
print("="*50 + "\n")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

while True:
    try:
        user_input = input("\033[1;34mUser:\033[0m ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})
        outputs = llm.chat(messages, sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text
        print(f"\n\033[1;32mAI:\033[0m {response_text}\n")
        messages.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        break
