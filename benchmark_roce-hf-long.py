import os
import sys
import time
import math
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
# d. Export your huggingface key on the first node: `export HF_TOKEN="<token>"`

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
#model_name = "meta-llama/Llama-3.3-70B-Instruct"
#model_name = "allura-forge/Llama-3.3-8B-Instruct"
model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"


print(f"🚀 Initializing {model_name}...")

try:
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,
        #tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.75, 
        
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
# Goal: Stress test for at least 10 minutes.
# Strategy: Use ignore_eos=True to force fixed-length generation.
# Calculation: Total Tokens = (Target Seconds) * (Est. Tokens/Sec Throughput)

TARGET_MINUTES = 12       # Aim for 12 mins to be safe
EST_TOKENS_PER_SEC = 5000 # Conservative estimate for dual GB10s (adjust if run is too short)
GEN_LEN = 4096            # Force 4k tokens per request

# Calculate how many prompts are needed to sustain this load
total_tokens_needed = TARGET_MINUTES * 60 * EST_TOKENS_PER_SEC
num_prompts_needed = math.ceil(total_tokens_needed / GEN_LEN)

print(f"\n📊 STRESS TEST CONFIGURATION:")
print(f"   Target Runtime:   {TARGET_MINUTES} minutes")
print(f"   Est. Throughput:  {EST_TOKENS_PER_SEC} tok/s")
print(f"   Tokens per req:   {GEN_LEN}")
print(f"   Total Prompts:    {num_prompts_needed}")
print(f"   Total Tokens:     {num_prompts_needed * GEN_LEN:,}")

# Create a long list of repeated prompts
base_prompts = [
    "Explain the theory of relativity in extreme detail.",
    "Write a Python script to train a neural network from scratch.",
    "Summarize the entire history of the Roman Empire.",
    "Describe the biological process of photosynthesis step-by-step.",
    "Write a science fiction story about a mission to Alpha Centauri."
]

# Multiply the list to reach the required number of prompts
repeat_factor = math.ceil(num_prompts_needed / len(base_prompts))
prompts = (base_prompts * repeat_factor)[:num_prompts_needed]

# ignore_eos=True is CRITICAL. It ensures the model does not stop generating 
# until it hits exactly max_tokens.
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95, 
    max_tokens=GEN_LEN,
    ignore_eos=True 
)

print(f"\n⚡ Starting Generation (This will take approx {TARGET_MINUTES} mins)...")
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
total_time = time.time() - start_time

# --- 6. RESULTS ---
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print("\n" + "="*50)
print(f"✅ SUCCESS")
print(f"Total Time: {total_time:.2f} s ({total_time/60:.2f} min)")
print(f"Throughput: {total_tokens / total_time:.2f} tokens/s")
print("="*50)