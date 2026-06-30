#ifndef PIPELINE_H
#define PIPELINE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include "fixedpoint.h"
#include "tflite_manager.h"
struct DynamicFCGraph {
    bool isValid = false;
    Qnn_ContextHandle_t contextHandle = nullptr;
    Qnn_GraphHandle_t graphHandle = nullptr;
    Qnn_Tensor_t tensorA;
    Qnn_Tensor_t tensorB;
    Qnn_Tensor_t tensorFC_Out;
    Qnn_Tensor_t tensorBias;
    std::vector<int32_t> biasData;
    std::vector<std::string> tensorNames;
    uint32_t dimsA[2];
    uint32_t dimsB[2];
    uint32_t dimsC[2];
    uint32_t dimsBias[1];
};
inline std::unordered_map<std::string, std::shared_ptr<DynamicFCGraph>> global_dynamic_fc_graphs;
inline std::mutex global_dynamic_fc_graphs_mutex;
inline void matmul_tiled(const float* A, const float* B, float* C,
                         int M, int K, int N,
                         int m_split, int k_split, int n_split,
                         int backend_choice = 1) {
    bool do_prof = std::getenv("PROF_PIPELINE") != nullptr || std::getenv("PROF_NPU") != nullptr;
    auto& engine = QnnHtaExecutionEngine::getInstance();
    engine.loadBackend(backend_choice);
    if (engine.isLoaded()) {
        int M_block = M / m_split;
        int K_block = K / k_split;
        int N_block = N / n_split;
        long long total_compile_us = 0;
        long long total_exec_us = 0;
        int compile_count = 0;
        int exec_count = 0;
        auto block_A = std::make_unique<FixedPointBlock<int16_t>>(M, K, K, true);
        auto block_B = std::make_unique<FixedPointBlock<int8_t>>(N, K, K, true);
        block_A->fit_exponent(A);
        block_A->floats_to_mantissa(A, 0, 8);
        std::vector<float> B_transposed(N * K);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < K; ++c) {
                B_transposed[r * K + c] = B[c * N + r];
            }
        }
        block_B->fit_exponent(B_transposed.data());
        block_B->floats_to_mantissa(B_transposed.data(), 0, 8);
        std::memset(C, 0, M * N * sizeof(float));
        std::string key = std::to_string(M_block) + "_" + std::to_string(K_block) + "_" + std::to_string(N_block);
        std::shared_ptr<DynamicFCGraph> graph_ptr;
        {
            std::lock_guard<std::mutex> lock(global_dynamic_fc_graphs_mutex);
            auto it = global_dynamic_fc_graphs.find(key);
            if (it != global_dynamic_fc_graphs.end()) {
                graph_ptr = it->second;
            } else {
                auto t_compile_start = std::chrono::high_resolution_clock::now();
                graph_ptr = std::make_shared<DynamicFCGraph>();
                auto* interface = engine.getInterface();
                auto& qnn = interface->QNN_INTERFACE_VER_NAME;
                qnn.contextCreate(engine.getBackendHandle(), engine.getDeviceHandle(), nullptr, &graph_ptr->contextHandle);
                qnn.graphCreate(graph_ptr->contextHandle, "dynamic_fc_graph", nullptr, &graph_ptr->graphHandle);
                graph_ptr->dimsA[0] = M_block;
                graph_ptr->dimsA[1] = K_block;
                graph_ptr->dimsB[0] = N_block;
                graph_ptr->dimsB[1] = K_block;
                graph_ptr->dimsC[0] = M_block;
                graph_ptr->dimsC[1] = N_block;
                graph_ptr->dimsBias[0] = N_block;
                std::memset(&graph_ptr->tensorA, 0, sizeof(Qnn_Tensor_t));
                graph_ptr->tensorA.version = QNN_TENSOR_VERSION_1;
                graph_ptr->tensorA.v1.id = 0;
                graph_ptr->tensorA.v1.name = "A";
                graph_ptr->tensorA.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
                graph_ptr->tensorA.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                graph_ptr->tensorA.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
                graph_ptr->tensorA.v1.rank = 2;
                graph_ptr->tensorA.v1.dimensions = graph_ptr->dimsA;
                graph_ptr->tensorA.v1.memType = QNN_TENSORMEMTYPE_RAW;
                graph_ptr->tensorA.v1.clientBuf.data = nullptr;
                graph_ptr->tensorA.v1.clientBuf.dataSize = 0;
                graph_ptr->tensorA.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
                graph_ptr->tensorA.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                graph_ptr->tensorA.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
                graph_ptr->tensorA.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
                qnn.tensorCreateGraphTensor(graph_ptr->graphHandle, &graph_ptr->tensorA);

                std::memset(&graph_ptr->tensorB, 0, sizeof(Qnn_Tensor_t));
                graph_ptr->tensorB.version = QNN_TENSOR_VERSION_1;
                graph_ptr->tensorB.v1.id = 1;
                graph_ptr->tensorB.v1.name = "B";
                graph_ptr->tensorB.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
                graph_ptr->tensorB.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                graph_ptr->tensorB.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
                graph_ptr->tensorB.v1.rank = 2;
                graph_ptr->tensorB.v1.dimensions = graph_ptr->dimsB;
                graph_ptr->tensorB.v1.memType = QNN_TENSORMEMTYPE_RAW;
                graph_ptr->tensorB.v1.clientBuf.data = nullptr;
                graph_ptr->tensorB.v1.clientBuf.dataSize = 0;
                graph_ptr->tensorB.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
                graph_ptr->tensorB.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                graph_ptr->tensorB.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
                graph_ptr->tensorB.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
                qnn.tensorCreateGraphTensor(graph_ptr->graphHandle, &graph_ptr->tensorB);

                std::memset(&graph_ptr->tensorBias, 0, sizeof(Qnn_Tensor_t));
                graph_ptr->tensorBias.version = QNN_TENSOR_VERSION_1;
                graph_ptr->tensorBias.v1.id = 2;
                graph_ptr->tensorBias.v1.name = "bias";
                graph_ptr->tensorBias.v1.type = QNN_TENSOR_TYPE_STATIC;
                graph_ptr->tensorBias.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                graph_ptr->tensorBias.v1.dataType = QNN_DATATYPE_SFIXED_POINT_32;
                graph_ptr->tensorBias.v1.rank = 1;
                graph_ptr->tensorBias.v1.dimensions = graph_ptr->dimsBias;
                graph_ptr->tensorBias.v1.memType = QNN_TENSORMEMTYPE_RAW;
                graph_ptr->biasData.assign(N_block, 0);
                graph_ptr->tensorBias.v1.clientBuf.data = graph_ptr->biasData.data();
                graph_ptr->tensorBias.v1.clientBuf.dataSize = N_block * sizeof(int32_t);
                graph_ptr->tensorBias.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
                graph_ptr->tensorBias.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                graph_ptr->tensorBias.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
                graph_ptr->tensorBias.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
                qnn.tensorCreateGraphTensor(graph_ptr->graphHandle, &graph_ptr->tensorBias);

                std::memset(&graph_ptr->tensorFC_Out, 0, sizeof(Qnn_Tensor_t));
                graph_ptr->tensorFC_Out.version = QNN_TENSOR_VERSION_1;
                graph_ptr->tensorFC_Out.v1.id = 3;
                graph_ptr->tensorFC_Out.v1.name = "C";
                graph_ptr->tensorFC_Out.v1.type = QNN_TENSOR_TYPE_APP_READ;
                graph_ptr->tensorFC_Out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                graph_ptr->tensorFC_Out.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
                graph_ptr->tensorFC_Out.v1.rank = 2;
                graph_ptr->tensorFC_Out.v1.dimensions = graph_ptr->dimsC;
                graph_ptr->tensorFC_Out.v1.memType = QNN_TENSORMEMTYPE_RAW;
                graph_ptr->tensorFC_Out.v1.clientBuf.data = nullptr;
                graph_ptr->tensorFC_Out.v1.clientBuf.dataSize = 0;
                graph_ptr->tensorFC_Out.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
                graph_ptr->tensorFC_Out.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                graph_ptr->tensorFC_Out.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
                graph_ptr->tensorFC_Out.v1.quantizeParams.scaleOffsetEncoding.offset = 32768;
                qnn.tensorCreateGraphTensor(graph_ptr->graphHandle, &graph_ptr->tensorFC_Out);
                Qnn_OpConfig_t opConfig;
                std::memset(&opConfig, 0, sizeof(opConfig));
                opConfig.version = QNN_OPCONFIG_VERSION_1;
                opConfig.v1.name = "fc_node";
                opConfig.v1.packageName = "qti.aisw";
                opConfig.v1.typeName = "FullyConnected";
                opConfig.v1.numOfParams = 0;
                opConfig.v1.params = nullptr;
                opConfig.v1.numOfInputs = 3;
                Qnn_Tensor_t inputs[3] = { graph_ptr->tensorA, graph_ptr->tensorB, graph_ptr->tensorBias };
                opConfig.v1.inputTensors = inputs;
                opConfig.v1.numOfOutputs = 1;
                opConfig.v1.outputTensors = &graph_ptr->tensorFC_Out;
                qnn.graphAddNode(graph_ptr->graphHandle, opConfig);
                Qnn_ErrorHandle_t finalizeStatus = qnn.graphFinalize(graph_ptr->graphHandle, nullptr, nullptr);
                if (finalizeStatus != QNN_SUCCESS) {
                    std::cerr << "ERROR: QnnGraph_finalize (Dynamic Weights) failed with code: 0x" << std::hex << finalizeStatus << std::dec << std::endl;
                    graph_ptr->isValid = false;
                } else {
                    graph_ptr->isValid = true;
                }
                auto t_compile_end = std::chrono::high_resolution_clock::now();
                total_compile_us += std::chrono::duration_cast<std::chrono::microseconds>(t_compile_end - t_compile_start).count();
                compile_count++;
                global_dynamic_fc_graphs[key] = graph_ptr;
            }
        }
        if (graph_ptr->isValid) {
            auto* interface = engine.getInterface();
            auto& qnn = interface->QNN_INTERFACE_VER_NAME;
            uint8_t* exec_out = (uint8_t*)Scratchpad::alloc(M_block * N_block * sizeof(uint16_t));
            for (int i = 0; i < k_split; ++i) {
                uint16_t* u_A_buf = (uint16_t*)Scratchpad::alloc(M_block * K_block * sizeof(uint16_t));
                uint8_t* u_B_buf = (uint8_t*)Scratchpad::alloc(N_block * K_block * sizeof(uint8_t));
                for (int r = 0; r < M_block; ++r) {
                    for (int c = 0; c < K_block; ++c) {
                        u_A_buf[r * K_block + c] = (uint16_t)(block_A->mantissa[r * K + i * K_block + c] + 127);
                    }
                }
                for (int r = 0; r < N_block; ++r) {
                    for (int c = 0; c < K_block; ++c) {
                        u_B_buf[r * K_block + c] = (uint8_t)(block_B->mantissa[r * K + i * K_block + c] + 127);
                    }
                }
                Qnn_Tensor_t exec_tensor_A = graph_ptr->tensorA;
                exec_tensor_A.v1.clientBuf.data = u_A_buf;
                exec_tensor_A.v1.clientBuf.dataSize = M_block * K_block * sizeof(uint16_t);
                Qnn_Tensor_t exec_tensor_B = graph_ptr->tensorB;
                exec_tensor_B.v1.clientBuf.data = u_B_buf;
                exec_tensor_B.v1.clientBuf.dataSize = N_block * K_block * sizeof(uint8_t);
                std::memset(exec_out, 0, M_block * N_block * sizeof(uint16_t));
                Qnn_Tensor_t exec_tensor_C = graph_ptr->tensorFC_Out;
                exec_tensor_C.v1.clientBuf.data = exec_out;
                exec_tensor_C.v1.clientBuf.dataSize = M_block * N_block * sizeof(uint16_t);
                Qnn_Tensor_t exec_inputs[2] = { exec_tensor_A, exec_tensor_B };
                auto t_exec_start = std::chrono::high_resolution_clock::now();
                qnn.graphExecute(graph_ptr->graphHandle, exec_inputs, 2, &exec_tensor_C, 1, nullptr, nullptr);
                auto t_exec_end = std::chrono::high_resolution_clock::now();
                total_exec_us += std::chrono::duration_cast<std::chrono::microseconds>(t_exec_end - t_exec_start).count();
                exec_count++;
                #pragma omp parallel for if(M_block*N_block > 4000)
                for (int r = 0; r < M_block; ++r) {
                    float mean_a = block_A->means[r];
                    int32_t exp_a = block_A->exponents[r];
                    int32_t prec_a = block_A->precisions[r];
                    for (int c = 0; c < N_block; ++c) {
                        int32_t total_exp = exp_a + block_B->exponents[c];
                        int32_t total_prec = prec_a + block_B->precisions[c];
                        float scale = fast_pow2(total_exp - total_prec);
                        uint16_t raw_val_u16 = ((uint16_t*)exec_out)[r * N_block + c];
                        float raw_val = (float)(static_cast<int32_t>(raw_val_u16) - 32768);
                        float dequantized = raw_val * scale;
                        float correction = (float)K_block * mean_a * block_B->means[c];
                        C[r * N + c] += dequantized + correction;
                    }
                }
                Scratchpad::free(u_A_buf);
                Scratchpad::free(u_B_buf);
            }
            Scratchpad::free(exec_out);
        } else {
            std::cerr << "ERROR: Graph compilation failed, bypassing execution entirely." << std::endl;
        }
        if (do_prof) {
            std::cout << "\n==================================================" << std::endl;
            std::cout << "NPU DETAILED PROFILING: " << M << "x" << K << "x" << N << " MATMUL" << std::endl;
            std::cout << "==================================================" << std::endl;
            if (compile_count > 0) {
                std::cout << "  Compilation Time (1 graph): " << (double)total_compile_us / 1000.0 << " ms" << std::endl;
            } else {
                std::cout << "  Compilation Time: 0 ms (Cached Dynamic Graph!)" << std::endl;
            }
            if (exec_count > 0) {
                std::cout << "  Average NPU Execution Time per Step (" << exec_count << " steps): " 
                          << (double)total_exec_us / exec_count / 1000.0 << " ms" << std::endl;
            }
            std::cout << "==================================================\n" << std::endl;
        }
    }
}
#endif