import os
import sys
import time
import torch
import shutil
import glob
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ["PROF_NPU"] = "1"
os.environ["PROF_PIPELINE"] = "0"
os.environ["NPUBLAS_QUIET"] = "1"
os.environ["NPUBLAS_VERBOSE"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["USE_STATIC_WEIGHTS"] = "0"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["KMP_AFFINITY"] = "disabled"
def configure_environment():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(project_dir, "libs")
    target_dir = "/data/local/tmp"
    if os.path.exists(libs_dir) and os.path.exists(target_dir):
        skel_files = glob.glob(os.path.join(libs_dir, "*Skel.so"))
        for skel in skel_files:
            target_path = os.path.join(target_dir, os.path.basename(skel))
            if os.path.exists(target_path):
                continue
            try:
                shutil.copy2(skel, target_dir)
                os.chmod(target_path, 0o755)
            except Exception:
                pass
import pytorch_npublas
def main():
    configure_environment()
    print("\n" + "="*50)
    print("NPU VS CPU MATRIX MULTIPLICATION BENCHMARK")
    print("="*50)
    print("Shape: 1024 x 1024 x 1024")
    print(f"Weight Mode: {'Static' if os.environ.get('USE_STATIC_WEIGHTS') == '1' else 'Dynamic'}")
    pytorch_npublas.set_backend(1)
    pytorch_npublas.set_power_mode(5)
    N = 1024
    A = torch.randn(N, N, dtype=torch.float32)
    B = torch.randn(N, N, dtype=torch.float32)
    print("\nPre-compiling and warming up NPU Graph...")
    pytorch_npublas.prepare_matmul(N, N, N, input_type=8, output_type=16)
    _ = torch.matmul(A, B, backend_choice=1)
    iterations = 5
    print(f"Running NPU Benchmark ({iterations} iterations)...")
    npu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C_npu = torch.matmul(A, B, backend_choice=1)
        npu_times.append((time.perf_counter() - start) * 1000)
    avg_npu_time = sum(npu_times) / iterations
    print(f"Running CPU Benchmark ({iterations} iterations)...")
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C_cpu = torch._matmul_original(A, B)
        cpu_times.append((time.perf_counter() - start) * 1000)
    avg_cpu_time = sum(cpu_times) / iterations
    abs_err = torch.abs(C_cpu - C_npu).mean().item()
    speedup = avg_cpu_time / avg_npu_time if avg_npu_time > 0 else 0.0
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Average CPU Time   : {avg_cpu_time:.2f} ms")
    print(f"Average NPU Time   : {avg_npu_time:.2f} ms")
    print(f"Speedup            : {speedup:.2f}x")
    print(f"Mean Absolute Error: {abs_err:.5f}")
    print("="*50 + "\n")
if __name__ == "__main__":
    main()
