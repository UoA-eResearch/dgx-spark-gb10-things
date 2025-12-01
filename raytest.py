import ray
import torch

# Connect to the local cluster
ray.init(address="auto")

print(f"Nodes in cluster: {len(ray.nodes())}")
print(f"Total GPUs: {ray.cluster_resources().get('GPU')}")

@ray.remote(num_gpus=1)
def check_gpu():
    # Verify we are on a Grace-Blackwell node
    return torch.cuda.get_device_name(0)

# Fire off tasks to both nodes
results = ray.get([check_gpu.remote() for _ in range(int(ray.cluster_resources().get('GPU')))])
print("Devices found:", results)
