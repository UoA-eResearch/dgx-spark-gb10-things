import os
import sys
import time
from vllm import LLM, SamplingParams

# --- 1. PREREQUISITES
# a. run in NVIDIA upstream vLLM container, pull to both nodes:
#    `sudo docker run --gpus all --network=host --privileged --shm-size=16g -v $(pwd):/workspace -it --env RAY_grpc_enable_http_proxy=0 --env no_proxy="localhost,127.0.0.1,192.168.100.1,192.168.100.2" --name ray_head nvcr.io/nvidia/vllm:25.10-py3 bash`
# b. set these environment variables on both nodes (in the vLLM container)
#    ```export VLLM_HOST_IP=192.168.100.1
#       export RAY_ADDRESS="192.168.100.1:6379"
#       export GLOO_SOCKET_IFNAME=enp1s0f0np0
#       export NCCL_SOCKET_IFNAME=enp1s0f0np0
#       export VLLM_USE_V1=0
#       export RAY_grpc_enable_http_proxy=0
#       export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,192.168.100.1,192.168.100.2"
#       export NO_PROXY=$no_proxy
#       export RAY_memory_monitor_refresh_ms=0
#       export RAY_memory_usage_threshold=1.0```
# c. Start the ray cluster on both nodes, in vLLM container
#    `ray start --head --port=6379 --node-ip-address=192.168.100.1` #first node
#    `ray start --address='192.168.100.1:6379' \
#          --node-ip-address=192.168.100.2 \
#          --num-gpus=1` # second node

# --- 2. AUTHENTICATION ---
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ Error: HF_TOKEN environment variable is not set.")
    sys.exit(1)

# --- 3. NETWORK CONFIGURATION ---
nccl_envs = {
    # Debug & Stability
    "NCCL_DEBUG": "INFO",
    "NCCL_COLLNET_ENABLE": "0",  # Disable Hardware Aggregation (Fixes Truncated Message)
    "NCCL_ALGO": "Ring",         # Force Ring Topology
    "NCCL_P2P_DISABLE": "1",     # Disable PCIe P2P (Use RoCE)
    "NCCL_IB_DISABLE": "0",      # Enable RoCE
    
    # Interface Config
    "NCCL_IB_HCA": "^mlx5",      # Match IB devices
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0,enp1s0f1np1",
    "NCCL_IB_GID_INDEX": "3",
    
    # Pass Token & Ray Configs to Workers
    "HF_TOKEN": hf_token,
    "RAY_memory_monitor_refresh_ms": "0",
    "RAY_memory_usage_threshold": "1.0",
}
os.environ.update(nccl_envs)

print("✅ Configured Ray to ignore High RAM usage (Unified Memory Fix).")

# --- 4. MODEL CONFIGURATION ---
model_name = "meta-llama/Llama-3.3-70B-Instruct"

print(f"🚀 Initializing {model_name}...")

try:
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.87, 
        
        # --- MEMORY SAFETY ---
        # We enable Eager Mode. This DISABLES CUDA Graphs.
        # CUDA Graphs consume extra memory during capture. 
        # Enabling eager mode reduces memory pressure significantly.
        enforce_eager=True, 
        
        max_model_len=8192,
        trust_remote_code=True,
    )
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)

# --- 5. BENCHMARK ---
prompts = [
    "Make a list 20 famous people who were named Sean."
    "Explain the theory of relativity.",
    "Write a Python script to sort a list.",
    "What is the capital of New Zealand?",
    "Summarize the history of the internet."
] * 5

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

print(f"\n⚡ Starting Generation...")
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
total_time = time.time() - start_time

# --- 6. RESULTS ---
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print("\n" + "="*50)
print(f"✅ SUCCESS")
print(f"Throughput: {total_tokens / total_time:.2f} tokens/s")
print("="*50)
