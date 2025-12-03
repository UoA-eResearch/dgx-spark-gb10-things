import os
import sys
import time
import ray
from vllm import LLM, SamplingParams

# --- STEP 1: CONFIGURATION ---
# To compare 1 vs 2 nodes, verify these settings:
# For 2 Nodes: tensor_parallel_size=2
# For 1 Node:  tensor_parallel_size=1 (and remove the ray connection if running locally)

TP_SIZE = 1  # <--- CHANGE THIS TO 1 TO TEST SINGLE NODE
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# --- ROCE SETUP (Keep existing config) ---
head_node_ip = "192.168.100.1"
nccl_envs = {
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_HCA": "^rocep1",
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0,enp1s0f1np1",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_RETRY_CNT": "7",
    "MAX_JOBS": "1",
    "FLASHINFER_MAX_JOBS": "1"
}
os.environ.update(nccl_envs)

# Only connect to Ray if we are distributed
if TP_SIZE > 1:
    print(f"🔗 Connecting to Ray Cluster (TP={TP_SIZE})...")
    ray.init(
        address=f"{head_node_ip}:6379",
        ignore_reinit_error=True,
        log_to_driver=False,
        _node_ip_address=head_node_ip,
        runtime_env={"env_vars": nccl_envs}
    )
else:
    print(f"💻 Running in Single-Node Mode (TP={TP_SIZE})...")

# --- INITIALIZE ---
print(f"🧠 Loading {MODEL_NAME}...")
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=TP_SIZE,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.70,
    enforce_eager=True,
    disable_log_stats=False # This prints the live vLLM stats to console
)

# Use greedy sampling for consistent speed tests (temperature=0)
sampling_params = SamplingParams(temperature=0, max_tokens=2048)

# --- THE TEST ---
print("\n" + "="*50)
print(f"🚀 STARTING BENCHMARK (TP={TP_SIZE})")
print("="*50 + "\n")

# A prompt that asks for a LONG output
prompts = [
    "Write a very detailed, 2000-word history of the Roman Empire, focusing on the fall of the republic.",
]

# Warmup (optional, helps stabilize buffers)
print("Warmup run...")
llm.generate(["Hello"], SamplingParams(max_tokens=10))

# Actual Run
start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()

# --- RESULTS ---
total_duration = end_time - start_time
generated_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
tps = generated_tokens / total_duration

print("\n" + "="*50)
print("📊 BENCHMARK RESULTS")
print("="*50)
print(f"Total Time:       {total_duration:.2f} s")
print(f"Tokens Generated: {generated_tokens}")
print(f"Speed:            \033[1;32m{tps:.2f} tokens/sec\033[0m")
print("="*50 + "\n")
