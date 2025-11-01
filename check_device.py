import torch

print("🔍 Checking PyTorch device availability...\n")

# CUDA availability
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU is available and will be used.")
    print(f"🧠 GPU Name: {device_name}")
    print(f"💽 CUDA Version: {torch.version.cuda}")
else:
    print("⚠️ GPU not available — training will run on CPU.")
    print("   (Ensure you have installed the correct CUDA-enabled PyTorch build)")

# General info
print("\nPyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
