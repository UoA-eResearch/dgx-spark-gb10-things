import os
import sys
import ray
from vllm import LLM, SamplingParams

# --- STEP 1: CLEANUP PROXIES (Still good to have in Python) ---
no_proxy_str = "localhost,127.0.0.1,0.0.0.0,::1,192.168.100.1,192.168.100.2"
os.environ["no_proxy"] = no_proxy_str
os.environ["NO_PROXY"] = no_proxy_str
os.environ["RAY_grpc_enable_http_proxy"] = "0"

# --- STEP 2: CONNECT ---
# We trust the shell variables now!
node_ip = "192.168.100.1"
ray_addr = f"{node_ip}:6379"

print(f"🔗 Connecting to Ray at {ray_addr}...")

ray.init(
    address=ray_addr,
    ignore_reinit_error=True,
    log_to_driver=False
    # Note: We REMOVED the runtime_env dictionary because
    # we exported everything globally in the shell.
)

# --- STEP 3: RUN WORKLOAD ---
print("🧠 Initializing vLLM...")

try:
    llm = LLM(
        model="facebook/opt-125m",
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        enforce_eager=True 
    )

    prompts = [
        "Name the most famous people in history with the name Sean",
        "Can you write me a python script that will get the openstack neutron router ip address for all tenants in a domain (domain=nz), and test pinging them to check connectivity?"
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    print("🚀 Generating text...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n--- RESULTS ---")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")

except Exception as e:
    print(f"\n❌ CRASH DETECTED: {e}")
