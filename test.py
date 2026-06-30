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
    print("\n=== STARTING N x N x N DIMENSIONS TEST SUITE (Quiet Mode) ===")
    print(f"Weight Mode: {'Static' if os.environ.get('USE_STATIC_WEIGHTS') == '1' else 'Dynamic'}")
    pytorch_npublas.set_backend(1)
    pytorch_npublas.set_power_mode(5)
    
    success_count = 0
    failure_count = 0
    for N in range(2, 65):
        A = torch.randn(N, N, dtype=torch.float32)
        B = torch.randn(N, N, dtype=torch.float32)
        try:
            pytorch_npublas.prepare_matmul(N, N, N, input_type=8, output_type=16)
            start = time.perf_counter()
            C_npu = torch.matmul(A, B, backend_choice=1)
            npu_time = (time.perf_counter() - start) * 1000
            C_cpu = torch._matmul_original(A, B)
            abs_err = torch.abs(C_cpu - C_npu).mean().item()
            print(f"  Shape {N}x{N}x{N}: SUCCESS | Time: {npu_time:.2f} ms | Mean Error: {abs_err:.5f}")
            success_count += 1
        except Exception as e:
            print(f"  Shape {N}x{N}x{N}: FAILED with error: {e}")
            failure_count += 1
    print("\n" + "="*40)
    print("TEST SUITE SUMMARY")
    print("="*40)
    print(f"Total Shapes Passed : {success_count} / 63")
    print(f"Total Shapes Failed : {failure_count} / 63")
    print("="*40)
if __name__ == "__main__":
    main()