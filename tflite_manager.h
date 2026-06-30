#ifndef TFLITE_MANAGER_H
#define TFLITE_MANAGER_H
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <chrono>
#include "fixedpoint.h"
#include "utils/execution.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"
struct Model {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    int M;
    int K;
    int N;
    int backend_choice = -1;
    TfLiteDelegate* delegate = nullptr;
    void* persist_a = nullptr;
    void* persist_b = nullptr;
    void* persist_out = nullptr;
    size_t size_a = 0;
    size_t size_b = 0;
    size_t size_out = 0;
    Model() = default;
    Model(std::unique_ptr<tflite::FlatBufferModel> m_model, std::unique_ptr<tflite::Interpreter> m_interp, int m_M, int m_K, int m_N, int m_backend, TfLiteDelegate* m_delegate)
        : model(std::move(m_model)), interpreter(std::move(m_interp)), M(m_M), K(m_K), N(m_N), backend_choice(m_backend), delegate(m_delegate) {}
    Model(Model&& other) noexcept {
        model = std::move(other.model);
        interpreter = std::move(other.interpreter);
        M = other.M;
        K = other.K;
        N = other.N;
        backend_choice = other.backend_choice;
        delegate = other.delegate;
        persist_a = other.persist_a;
        persist_b = other.persist_b;
        persist_out = other.persist_out;
        size_a = other.size_a;
        size_b = other.size_b;
        size_out = other.size_out;
        other.delegate = nullptr;
        other.persist_a = nullptr;
        other.persist_b = nullptr;
        other.persist_out = nullptr;
    }
    Model& operator=(Model&& other) noexcept {
        if (this != &other) {
            interpreter.reset();
            if (delegate) {
                TfLiteExternalDelegateDelete(delegate);
            }
            if (persist_a) {
                Scratchpad::free(persist_a);
            }
            if (persist_b) {
                Scratchpad::free(persist_b);
            }
            if (persist_out) {
                Scratchpad::free(persist_out);
            }
            model = std::move(other.model);
            interpreter = std::move(other.interpreter);
            M = other.M;
            K = other.K;
            N = other.N;
            backend_choice = other.backend_choice;
            delegate = other.delegate;
            persist_a = other.persist_a;
            persist_b = other.persist_b;
            persist_out = other.persist_out;
            size_a = other.size_a;
            size_b = other.size_b;
            size_out = other.size_out;
            other.delegate = nullptr;
            other.persist_a = nullptr;
            other.persist_b = nullptr;
            other.persist_out = nullptr;
        }
        return *this;
    }
    ~Model() {
        interpreter.reset();
        if (delegate) {
            TfLiteExternalDelegateDelete(delegate);
        }
        if (persist_a) {
            Scratchpad::free(persist_a);
        }
        if (persist_b) {
            Scratchpad::free(persist_b);
        }
        if (persist_out) {
            Scratchpad::free(persist_out);
        }
    }
};
inline std::vector<Model>* models = new std::vector<Model>();
inline void request_tflite_generation(int M, int K, int N, int backend_choice) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return;
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(9999);
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        close(sock);
        return;
    }
    struct timeval tv;
    tv.tv_sec = 15;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        close(sock);
        return;
    }
    std::string cmd = "generate " + std::to_string(M) + " " + std::to_string(K) + " " + std::to_string(N) + " " + std::to_string(backend_choice) + "\n";
    send(sock, cmd.c_str(), cmd.length(), 0);
    char buffer[1024] = {0};
    [[maybe_unused]] int valread = read(sock, buffer, 1024);
    close(sock);
}
inline Model create_model(int M, int K, int N, int backend_choice) {
    std::cerr << "[create_model] M=" << M << ", K=" << K << ", N=" << N << ", backend_choice=" << backend_choice << std::endl;
    request_tflite_generation(M, K, N, backend_choice);
    std::string filename;
    if (backend_choice == 3) {
        filename = "matmuls/matmul_fp32_" + std::to_string(M) + "x" + std::to_string(K) + "_" + std::to_string(K) + "x" + std::to_string(N) + ".tflite";
    } else {
        filename = "matmuls/matmul_a16w8_" + std::to_string(M) + "x" + std::to_string(K) + "_" + std::to_string(K) + "x" + std::to_string(N) + ".tflite";
    }
    auto model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());
    if (!model) {
        std::cerr << "ERROR: Failed to load FlatBufferModel from " << filename << std::endl;
        return Model{nullptr, nullptr, M, K, N, backend_choice, nullptr};
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    TfLiteDelegate* delegate = nullptr;
    if (backend_choice > 0 && interpreter) {
        char* home_env = std::getenv("HOME");
        std::string project_libs_dir = home_env ? (std::string(home_env) + "/projects/pytorch_npublas/libs") : "libs";
        std::string delegate_lib_path = project_libs_dir + "/libQnnTFLiteDelegate.so";
        std::string adsp_path = project_libs_dir + ";/data/local/tmp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
        setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), 1);
        std::string ld_path = project_libs_dir + ":" + (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "");
        setenv("LD_LIBRARY_PATH", ld_path.c_str(), 1);
        TfLiteExternalDelegateOptions options = TfLiteExternalDelegateOptionsDefault(delegate_lib_path.c_str());
        std::string backend_type_string;
        std::string library_path_string;
        if (backend_choice == 1) {
            backend_type_string = "htp";
            library_path_string = project_libs_dir + "/libQnnHtp.so";
        } else if (backend_choice == 2) {
            backend_type_string = "dsp";
            library_path_string = project_libs_dir + "/libQnnDsp.so";
        } else if (backend_choice == 3) {
            backend_type_string = "gpu";
            library_path_string = project_libs_dir + "/libQnnGpu.so";
        }
        options.insert(&options, "backend_type", backend_type_string.c_str());
        options.insert(&options, "library_path", library_path_string.c_str());
        delegate = TfLiteExternalDelegateCreate(&options);
        if (delegate) {
            if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
                std::cerr << "WARN: Failed to modify graph with QNN delegate for backend: " << backend_type_string << std::endl;
                TfLiteExternalDelegateDelete(delegate);
                delegate = nullptr;
            } else {
                std::cerr << "INFO: Successfully modified graph with QNN delegate for backend: " << backend_type_string << std::endl;
            }
        } else {
            std::cerr << "WARN: Failed to create external QNN delegate for backend: " << backend_type_string << std::endl;
        }
    }
    void* persist_a = nullptr;
    void* persist_b = nullptr;
    void* persist_out = nullptr;
    size_t size_a = 0;
    size_t size_b = 0;
    size_t size_out = 0;
    if (interpreter) {
        int input_a_idx = interpreter->inputs()[0];
        int input_b_idx = interpreter->inputs()[1];
        int output_idx = interpreter->outputs()[0];
        size_t req_bytes_a = interpreter->tensor(input_a_idx)->bytes;
        size_t req_bytes_b = interpreter->tensor(input_b_idx)->bytes;
        size_t req_bytes_out = interpreter->tensor(output_idx)->bytes;
        size_t type_size = (backend_choice == 3) ? sizeof(float) : sizeof(int16_t);
        size_a = std::max(req_bytes_a, M * K * type_size);
        size_b = std::max(req_bytes_b, K * N * type_size);
        size_out = req_bytes_out;
        persist_a = Scratchpad::alloc(size_a);
        persist_b = Scratchpad::alloc(size_b);
        persist_out = Scratchpad::alloc(size_out);
        TfLiteCustomAllocation alloc_a = { persist_a, size_a };
        interpreter->SetCustomAllocationForTensor(input_a_idx, alloc_a);
        TfLiteCustomAllocation alloc_b = { persist_b, size_b };
        interpreter->SetCustomAllocationForTensor(input_b_idx, alloc_b);
        TfLiteCustomAllocation alloc_out = { persist_out, size_out };
        interpreter->SetCustomAllocationForTensor(output_idx, alloc_out);
        interpreter->AllocateTensors();
    }
    Model model_res(std::move(model), std::move(interpreter), M, K, N, backend_choice, delegate);
    model_res.persist_a = persist_a;
    model_res.persist_b = persist_b;
    model_res.persist_out = persist_out;
    model_res.size_a = size_a;
    model_res.size_b = size_b;
    model_res.size_out = size_out;
    return model_res;
}
inline Model& fetch_model(int M, int K, int N, int backend_choice) {
    for (size_t i = 0; i < models->size(); ++i) {
        if ((*models)[i].M == M && (*models)[i].K == K && (*models)[i].N == N && (*models)[i].backend_choice == backend_choice) {
            return (*models)[i];
        }
    }
    models->push_back(create_model(M, K, N, backend_choice));
    return models->back();
}
inline void matmul_int8(const float* A, const float* B, float* C, int M, int K, int N, bool transposeA, bool transposeB, int backend_choice = 1) {
    bool do_prof = std::getenv("PROF_NPU") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();

    auto& engine = QnnHtaExecutionEngine::getInstance();
    engine.loadBackend(backend_choice);
    if (engine.isLoaded()) {
        QnnCachedGraph& graph = engine.getOrCreateGraph(M, K, N);
        if (!graph.isValid) {
            return;
        }
        
        FixedPointBlock<int8_t> fixed_A(M, K, K, true);
        fixed_A.fit_exponent(A);
        fixed_A.floats_to_mantissa(A, 0, 8);
        
        FixedPointBlock<int8_t> fixed_B(N, K, K, true);
        float* slice_B = new float[N * K];
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < K; ++c) {
                slice_B[r * K + c] = B[c * N + r];
            }
        }
        fixed_B.fit_exponent(slice_B);
        fixed_B.floats_to_mantissa(slice_B, 0, 8);
        delete[] slice_B;
        
        uint8_t* u_A = (uint8_t*)Scratchpad::alloc(M * K);
        uint8_t* u_B = (uint8_t*)Scratchpad::alloc(N * K);
        for (int i = 0; i < M * K; ++i) {
            u_A[i] = (uint8_t)(fixed_A.mantissa[i] + 127);
        }
        for (int i = 0; i < N * K; ++i) {
            u_B[i] = (uint8_t)(fixed_B.mantissa[i] + 127);
        }
        uint8_t* out_buf = (uint8_t*)Scratchpad::alloc(M * N * 2);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        engine.execute(graph, u_A, u_B, out_buf);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        Scratchpad::free(u_A);
        Scratchpad::free(u_B);

        #pragma omp parallel for if(M*N > 10000)
        for (int i = 0; i < M; ++i) {
            float mean_a = fixed_A.means[i];
            int32_t exp_a = fixed_A.exponents[i];
            int32_t prec_a = fixed_A.precisions[i];
            #pragma clang loop vectorize(enable)
            for (int j = 0; j < N; ++j) {
                int32_t total_exp = exp_a + fixed_B.exponents[j];
                int32_t total_prec = prec_a + fixed_B.precisions[j];
                float scale = fast_pow2(total_exp - total_prec);
                uint16_t raw_val_u16 = ((uint16_t*)out_buf)[i * N + j];
                float raw_val = (float)(static_cast<int32_t>(raw_val_u16) - 32768);
                float dequantized = raw_val * scale;
                float correction = (float)K * mean_a * fixed_B.means[j];
                C[i * N + j] = dequantized + correction;
            }
        }
        Scratchpad::free(out_buf);
        auto t3 = std::chrono::high_resolution_clock::now();
        if (do_prof) {
            auto dur_pre = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            auto dur_dsp = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            auto dur_post = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            std::cerr << "[PROF RAW QNN DIRECT] Pre-proc: " << dur_pre << " us, QNN Run: " << dur_dsp << " us, Post-proc: " << dur_post << " us" << std::endl;
        }
        return;
    }

    Model& model_raw = fetch_model(M, K, N, backend_choice);
    tflite::Interpreter* interpreter = model_raw.interpreter.get();
    if (!interpreter) {
        return;
    }
    if (backend_choice == 3) {
        std::memcpy(model_raw.persist_a, A, M * K * sizeof(float));
        std::memcpy(model_raw.persist_b, B, K * N * sizeof(float));
        if (interpreter->Invoke() != kTfLiteOk) {
            return;
        }
        std::memcpy(C, model_raw.persist_out, M * N * sizeof(float));
        return;
    }
    int output_idx = interpreter->outputs()[0];
    int target_precision = 8;
    FixedPointBlock<int16_t> fixed_A(M, K, K, true);
    fixed_A.fit_exponent(A);
    fixed_A.floats_to_mantissa(A, 0, target_precision);
    FixedPointBlock<int16_t>* fixed_B_ptr = nullptr;
    FixedPointBlock<int16_t>* popped = prefetch_queue.pop();
    if (popped) {
        if (popped->rows == K && popped->cols == N) {
            fixed_B_ptr = popped;
        } else {
            delete popped;
        }
    }
    bool used_prefetched = (fixed_B_ptr != nullptr);
    if (!fixed_B_ptr) {
        fixed_B_ptr = new FixedPointBlock<int16_t>(K, N, K, false);
        fixed_B_ptr->fit_exponent(B);
        fixed_B_ptr->floats_to_mantissa(B, 0, target_precision);
    }
    std::memcpy(model_raw.persist_a, fixed_A.mantissa, M * K * sizeof(int16_t));
    std::memcpy(model_raw.persist_b, fixed_B_ptr->mantissa, K * N * sizeof(int16_t));
    auto t1 = std::chrono::high_resolution_clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
        if (!used_prefetched) delete fixed_B_ptr;
        return;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    TfLiteType output_type = interpreter->tensor(output_idx)->type;
    if (output_type == kTfLiteInt16) {
        int16_t* output = (int16_t*)model_raw.persist_out;
        #pragma omp parallel for if(M*N > 10000)
        for (int i = 0; i < M; ++i) {
            float mean_a = fixed_A.means[i];
            int32_t exp_a = fixed_A.exponents[i];
            int32_t prec_a = fixed_A.precisions[i];
            #pragma clang loop vectorize(enable)
            for (int j = 0; j < N; ++j) {
                int32_t total_exp = exp_a + fixed_B_ptr->exponents[j];
                int32_t total_prec = prec_a + fixed_B_ptr->precisions[j];
                union {
                    int32_t i_val;
                    float f_val;
                } u;
                u.i_val = (127 + (total_exp - total_prec)) << 23;
                float scale = u.f_val;
                float dequantized_centered = (float)output[i * N + j] * scale;
                float correction = (float)K * mean_a * fixed_B_ptr->means[j];
                C[i * N + j] = dequantized_centered + correction;
            }
        }
    } else {
        int32_t* output = (int32_t*)model_raw.persist_out;
        #pragma omp parallel for if(M*N > 10000)
        for (int i = 0; i < M; ++i) {
            float mean_a = fixed_A.means[i];
            int32_t exp_a = fixed_A.exponents[i];
            int32_t prec_a = fixed_A.precisions[i];
            #pragma clang loop vectorize(enable)
            for (int j = 0; j < N; ++j) {
                int32_t total_exp = exp_a + fixed_B_ptr->exponents[j];
                int32_t total_prec = prec_a + fixed_B_ptr->precisions[j];
                union {
                    int32_t i_val;
                    float f_val;
                } u;
                u.i_val = (127 + (total_exp - total_prec)) << 23;
                float scale = u.f_val;
                float dequantized_centered = (float)output[i * N + j] * scale;
                float correction = (float)K * mean_a * fixed_B_ptr->means[j];
                C[i * N + j] = dequantized_centered + correction;
            }
        }
    }
    if (!used_prefetched) {
        delete fixed_B_ptr;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    if (do_prof) {
        auto dur_pre = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto dur_dsp = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto dur_post = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        std::cerr << "[DEPRECATED TFLITE FALLBACK PROF] Pre-proc: " << dur_pre << " us, CDSP Run: " << dur_dsp << " us, Post-proc: " << dur_post << " us" << std::endl;
    }
}
#endif