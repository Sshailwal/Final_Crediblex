import torch

print("GPU Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    # avoid "version is not a known attribute" by using getattr
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    print("CUDA Version:", cuda_version or "N/A")
else:
    print("GPU Name: N/A")
