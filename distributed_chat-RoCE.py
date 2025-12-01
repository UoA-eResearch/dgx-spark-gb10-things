import os
import sys
import ray
from vllm import LLM, SamplingParams

# --- STEP 1: DEFINE VARIABLES ---
head_node_ip = "192.168.100.1"

# The specific config for your hardware
# matches your 'ibdev2netdev' output: rocep1s0f0 / rocep1s0f1
nccl_envs = {
    # 1. Force RoCE v2 (UDP)
    "NCCL_IB_GID_INDEX": "3",
    
    # 2. Select specific interfaces (Regex to match both rocep1s0f0 and rocep1s0f1)
    "NCCL_IB_HCA": "^rocep1",
    
    # 3. Handshake on the 200Gb ethernet IP interfaces
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0,enp1s0f1np1",
    
    # 4. Enable RDMA
    "NCCL_IB_DISABLE": "0",
    
    # 5. Optimization for direct-connect (retries)
    "NCCL_IB_RETRY_CNT": "7",
    
    # 6. Debugging
    "NCCL_DEBUG": "INFO",
    
    # 7. System Limits (Prevent Exit 137)
    "MAX_JOBS": "1",
    "FLASHINFER_MAX_JOBS": "1"
}

# Apply them locally (for the Head Node)
os.environ.update(nccl_envs)

# --- STEP 2: CONNECT TO CLUSTER WITH RUNTIME ENV ---
ray_addr = f"{head_node_ip}:6379"
print(f"🔗 Connecting to Ray at {ray_addr}...")

# !!! THIS IS THE FIX !!!
# We pass 'runtime_env' to force these variables onto the remote Worker Node
ray.init(
    address=ray_addr, 
    ignore_reinit_error=True, 
    log_to_driver=False, 
    _node_ip_address=head_node_ip,
    runtime_env={"env_vars": nccl_envs}
)

# --- STEP 3: INITIALIZE MODEL ---
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"🧠 Loading {model_name}...")
print("   Mode: Distributed RoCE v2 (Dual Rail)")

llm = LLM(
    model=model_name,
    tensor_parallel_size=2,
    trust_remote_code=True,
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
