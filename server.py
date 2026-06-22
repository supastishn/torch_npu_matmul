import os
import socket
import numpy as np
import tensorflow as tf
def generate_tflite(M, K, N, is_gpu=False):
    os.makedirs("matmuls", exist_ok=True)
    if is_gpu:
        filename = f"matmuls/matmul_fp32_{M}x{K}_{K}x{N}.tflite"
    else:
        filename = f"matmuls/matmul_a16w8_{M}x{K}_{K}x{N}.tflite"
    if os.path.exists(filename):
        print(f"Loading cached model: {filename}")
        return filename
    print(f"Generating dynamic {'FP32' if is_gpu else 'W8A16'} TFLite model: {M}x{K}x{N}")
    class MatMulModel(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[M, K], dtype=tf.float32, name="A"),
            tf.TensorSpec(shape=[K, N], dtype=tf.float32, name="B")
        ])
        def __call__(self, A, B):
            return tf.matmul(A, B)
    try:
        model = MatMulModel()
        concrete_func = model.__call__.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        if not is_gpu:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            def rep_data():
                yield [np.random.uniform(-1, 1, size=(M, K)).astype(np.float32),
                       np.random.uniform(-1, 1, size=(K, N)).astype(np.float32)]
            converter.representative_dataset = rep_data
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        tflite_model = converter.convert()
        with open(filename, "wb") as f:
            f.write(tflite_model)
        print(f"Successfully generated: {filename}")
    except Exception as e:
        print(f"Conversion Error: {e}")
        raise e
    return filename
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 9999))
    server.listen(5)
    print("TFLite Generation Server listening on port 9999...")
    while True:
        try:
            conn, addr = server.accept()
            data = conn.recv(1024).decode('utf-8')
            if data.startswith('generate'):
                parts = data.strip().split()
                if len(parts) >= 4:
                    M, K, N = int(parts[1]), int(parts[2]), int(parts[3])
                    is_gpu = False
                    if len(parts) == 5 and int(parts[4]) == 3:
                        is_gpu = True
                    generate_tflite(M, K, N, is_gpu)
                    conn.sendall(b"OK\n")
            conn.close()
        except Exception as e:
            print(f"Error handling connection: {e}")
if __name__ == '__main__':
    start_server()