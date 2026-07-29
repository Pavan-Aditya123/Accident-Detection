import sys
import torch

def check_hardware_and_environment():
    print("=" * 60)
    print("      HARDWARE & ENVIRONMENT VERIFICATION SCRIPT")
    print("=" * 60)
    
    # 1. Python Version
    python_version = sys.version
    print(f"[+] Python Version       : {python_version.split()[0]}")
    print(f"    Full Version Info    : {python_version}")
    
    # 2. PyTorch Version
    pytorch_version = torch.__version__
    print(f"[+] PyTorch Version      : {pytorch_version}")
    
    # 3. CUDA Available Status
    cuda_available = torch.cuda.is_available()
    print(f"[+] CUDA Available       : {cuda_available}")
    
    # 4. PyTorch CUDA Compatibility & GPU Info
    if cuda_available:
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[+] PyTorch CUDA Version : {cuda_version}")
        print(f"[+] GPU Count            : {device_count}")
        print(f"[+] Primary GPU Name     : {gpu_name}")
    else:
        print("[!] CUDA is NOT available. PyTorch is running on CPU.")
        print("    If a GPU is installed, check NVIDIA Drivers & CUDA PyTorch installation.")

    print("=" * 60)

if __name__ == "__main__":
    check_hardware_and_environment()
