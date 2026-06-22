import time
import torch
import os
os.environ["PROF_NPU"] = "1"
import pytorch_npublas
import shutil
import glob
def setup_skel_libraries():
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
            except Exception as e:
                print(f"Warning: Failed to copy {skel} to {target_dir}: {e}")
def check_dsp_libraries():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(project_dir, "libs")
    missing_files = []
    required_dsp_files = [
        "libQnnDsp.so",
        "libQnnDspV66Stub.so",
        "libQnnDspV66Skel.so"
    ]
    for f in required_dsp_files:
        path = os.path.join(libs_dir, f)
        if not os.path.exists(path):
            missing_files.append(f)
    if missing_files:
        print("\n" + "="*70)
        print("DIAGNOSTIC WARNING: MISSING SNAPDRAGON 870 DSP v66 LIBRARIES")
        print("="*70)
        print("Your Snapdragon 870 chipset has a Hexagon v66 CDSP (not the newer HTP).")
        print("To run the DSP backend, the QNN SDK requires the v66 libraries:")
        for mf in missing_files:
            print(f"  - Missing: {mf}")
        print("\nHow to fix:")
        print("1. Locate these files in your Qualcomm QNN SDK:")
        print("   - libQnnDsp.so and libQnnDspV66Stub.so are located under:")
        print("     <QNN_SDK_ROOT>/lib/aarch64-android/")
        print("   - libQnnDspV66Skel.so is located under:")
        print("     <QNN_SDK_ROOT>/lib/hexagon-v66/unsigned/")
        print("2. Copy them to your project's 'libs' folder.")
        print("3. Ensure libQnnDspV66Skel.so is copied to /data/local/tmp:")
        print("   cp libs/libQnnDspV66Skel.so /data/local/tmp/")
        print("="*70 + "\n")
def benchmark_single_matmul():
    print("\n=== RUNNING SINGLE MATMUL BENCHMARK (1024x1024) ===")
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    start = time.perf_counter()
    C_cpu = torch._matmul_original(A, B)
    cpu_time = (time.perf_counter() - start) * 1000
    print(f"Standard CPU Matmul: {cpu_time:.2f} ms")
    print("\n[DSP Cold Run]")
    torch.matmul(A, B, backend_choice=2)
    print("\n[DSP Hot Run]")
    start = time.perf_counter()
    C_dsp = torch.matmul(A, B, backend_choice=2)
    dsp_time = (time.perf_counter() - start) * 1000
    print(f"DSP Accelerated Matmul: {dsp_time:.2f} ms")
    error = torch.abs(C_cpu - C_dsp).mean().item()
    print(f"Mean Absolute Error: {error:.4f}")
def benchmark_neural_network():
    print("\n=== RUNNING 3-LAYER NEURAL NETWORK FORWARD PASS ===")
    batch_size = 256
    features_in = 512
    hidden1 = 1024
    hidden2 = 512
    classes = 256
    x = torch.randn(batch_size, features_in, dtype=torch.float32)
    w1 = torch.randn(features_in, hidden1, dtype=torch.float32)
    w2 = torch.randn(hidden1, hidden2, dtype=torch.float32)
    w3 = torch.randn(hidden2, classes, dtype=torch.float32)
    start = time.perf_counter()
    h1_cpu = torch.relu(torch._matmul_original(x, w1))
    h2_cpu = torch.relu(torch._matmul_original(h1_cpu, w2))
    out_cpu = torch._matmul_original(h2_cpu, w3)
    cpu_total = (time.perf_counter() - start) * 1000
    print(f"Normal CPU NN Forward Pass: {cpu_total:.2f} ms")
    torch.matmul(x, w1, backend_choice=2)
    torch.matmul(h1_cpu, w2, backend_choice=2)
    torch.matmul(h2_cpu, w3, backend_choice=2)
    print("\n[DSP Hot Runs with Microsecond Profiling]")
    start = time.perf_counter()
    h1_dsp = torch.relu(torch.matmul(x, w1, backend_choice=2))
    h2_dsp = torch.relu(torch.matmul(h1_dsp, w2, backend_choice=2))
    out_dsp = torch.matmul(h2_dsp, w3, backend_choice=2)
    dsp_total = (time.perf_counter() - start) * 1000
    print(f"DSP Accelerated NN Forward Pass: {dsp_total:.2f} ms")
    error = torch.abs(out_cpu - out_dsp).mean().item()
    print(f"NN Output Mean Absolute Error: {error:.4f}")
if __name__ == "__main__":
    check_dsp_libraries()
    setup_skel_libraries()
    benchmark_single_matmul()
    benchmark_neural_network()