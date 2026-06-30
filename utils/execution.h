#ifndef UTILS_EXECUTION_H
#define UTILS_EXECUTION_H
#include <dlfcn.h>
#include <qualcomm/QNN/QnnBackend.h>
#include <qualcomm/QNN/QnnContext.h>
#include <qualcomm/QNN/QnnGraph.h>
#include <qualcomm/QNN/QnnTensor.h>
#include <qualcomm/QNN/QnnLog.h>
#include "qnn_hta_direct.h"
#include <qualcomm/QNN/QnnMem.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <chrono>
typedef void* (*RpcMemAllocFn_t)(int, uint32_t, int);
typedef void (*RpcMemFreeFn_t)(void*);
typedef int (*RpcMemToFdFn_t)(void*);
typedef void (*RpcMemInitFn_t)(void);
inline RpcMemAllocFn_t global_rpcmem_alloc = nullptr;
inline RpcMemFreeFn_t global_rpcmem_free = nullptr;
inline RpcMemToFdFn_t global_rpcmem_to_fd = nullptr;
inline bool rpcmem_loaded = false;
inline void load_rpcmem_once() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        void* handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            handle = dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (handle) {
            global_rpcmem_alloc = (RpcMemAllocFn_t)dlsym(handle, "rpcmem_alloc");
            global_rpcmem_free = (RpcMemFreeFn_t)dlsym(handle, "rpcmem_free");
            global_rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(handle, "rpcmem_to_fd");
            auto rpcmem_init = (RpcMemInitFn_t)dlsym(handle, "rpcmem_init");
            if (rpcmem_init) {
                rpcmem_init();
            }
            if (global_rpcmem_alloc && global_rpcmem_free) {
                rpcmem_loaded = true;
            }
        }
    });
}
typedef enum {
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_POWER_MODE = 1
} CustomQnnHtaPerfInfrastructure_PowerConfigOption_t;
typedef enum {
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_DEFAULT = 0,
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_LOW_POWER_SAVER = 1,
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER = 2,
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_HIGH_POWER_SAVER = 3,
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_DEFAULT_HTP = 4,
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_HIGH_PERFORMANCE = 5,
  CUSTOM_QNN_HTA_PERF_INFRASTRUCTURE_POWERMODE_BURST = 6
} CustomQnnHtaPerfInfrastructure_PowerMode_t;
struct QnnCachedGraph {
    bool isValid = false;
    Qnn_GraphHandle_t graphHandle;
    Qnn_ContextHandle_t contextHandle;
    Qnn_Tensor_t tensorA;
    Qnn_Tensor_t tensorB;
    Qnn_Tensor_t tensorC;
    Qnn_Tensor_t tensorBias;
    uint32_t dimsA[2];
    uint32_t dimsB[2];
    uint32_t dimsC[2];
    uint32_t dimsBias[1];
    std::vector<int32_t> biasData;
    std::vector<uint8_t> staticWeights;
    ~QnnCachedGraph();
};
inline void qnnLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
}
class QnnHtaExecutionEngine {
private:
    QnnLoadedBackendContainer backendContainer;
    Qnn_BackendHandle_t backendHandle;
    Qnn_DeviceHandle_t deviceHandle;
    Qnn_LogHandle_t logHandle;
    std::unordered_map<std::string, std::unique_ptr<QnnCachedGraph>> graphCache;
    bool diagnosticsPerformed;
    int currentBackendChoice;
    QnnMem_RegisterFn_t qnnMemRegister;
    QnnMem_DeRegisterFn_t qnnMemDeRegister;
    QnnHtaExecutionEngine() : backendContainer{nullptr, nullptr}, backendHandle(nullptr), deviceHandle(nullptr), logHandle(nullptr), diagnosticsPerformed(false), currentBackendChoice(-1), qnnMemRegister(nullptr), qnnMemDeRegister(nullptr) {
        loadBackend(1);
    }
public:
    static QnnHtaExecutionEngine& getInstance() {
        static QnnHtaExecutionEngine* instance = new QnnHtaExecutionEngine();
        return *instance;
    }
    ~QnnHtaExecutionEngine() {
        unloadActiveBackend();
    }
    void unloadActiveBackend() {
        if (backendContainer.interfaceProviderTable) {
            auto& qnn = backendContainer.interfaceProviderTable->QNN_INTERFACE_VER_NAME;
            for (auto& pair : graphCache) {
                if (pair.second) {
                    qnn.contextFree(pair.second->contextHandle, nullptr);
                }
            }
            graphCache.clear();
            if (deviceHandle && qnn.deviceFree) {
                qnn.deviceFree(deviceHandle);
                deviceHandle = nullptr;
            }
            if (backendHandle) {
                qnn.backendFree(backendHandle);
                backendHandle = nullptr;
            }
            if (logHandle && qnn.logFree) {
                qnn.logFree(logHandle);
                logHandle = nullptr;
            }
        }
        unloadQnnBackendSharedLibrary(backendContainer);
        qnnMemRegister = nullptr;
        qnnMemDeRegister = nullptr;
    }
    void loadBackend(int backendChoice) {
        if (currentBackendChoice == backendChoice && backendHandle != nullptr) {
            return;
        }
        unloadActiveBackend();
        currentBackendChoice = backendChoice;
        char* homeEnv = std::getenv("HOME");
        std::string projectLibsDir = homeEnv ? (std::string(homeEnv) + "/projects/pytorch_npublas/libs") : "libs";
        std::string adspPath = projectLibsDir + ";/data/local/tmp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
        setenv("ADSP_LIBRARY_PATH", adspPath.c_str(), 1);
        std::string ldPath = projectLibsDir + ":" + (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "");
        setenv("LD_LIBRARY_PATH", ldPath.c_str(), 1);
        std::string libraryPath;
        if (backendChoice == 1 || backendChoice == 7) libraryPath = projectLibsDir + "/libQnnHta.so";
        else if (backendChoice == 2) libraryPath = projectLibsDir + "/libQnnDsp.so";
        else if (backendChoice == 3) libraryPath = projectLibsDir + "/libQnnHtp.so";
        else if (backendChoice == 4) libraryPath = projectLibsDir + "/libQnnGpu.so";
        else libraryPath = projectLibsDir + "/libQnnHta.so";
        backendContainer = loadQnnBackendSharedLibrary(libraryPath);
        if (!backendContainer.librarySharedObjectHandle || !backendContainer.interfaceProviderTable) {
            std::string altPath;
            if (backendChoice == 1 || backendChoice == 7) altPath = "/data/local/tmp/libQnnHta.so";
            else if (backendChoice == 2) altPath = "/data/local/tmp/libQnnDsp.so";
            else if (backendChoice == 3) altPath = "/data/local/tmp/libQnnHtp.so";
            else if (backendChoice == 4) altPath = "/data/local/tmp/libQnnGpu.so";
            else altPath = "/data/local/tmp/libQnnHta.so";
            backendContainer = loadQnnBackendSharedLibrary(altPath);
        }
        if (backendContainer.librarySharedObjectHandle) {
            qnnMemRegister = reinterpret_cast<QnnMem_RegisterFn_t>(dlsym(backendContainer.librarySharedObjectHandle, "QnnMem_register"));
            qnnMemDeRegister = reinterpret_cast<QnnMem_DeRegisterFn_t>(dlsym(backendContainer.librarySharedObjectHandle, "QnnMem_deRegister"));
        }
        if (backendContainer.interfaceProviderTable) {
            auto& qnn = backendContainer.interfaceProviderTable->QNN_INTERFACE_VER_NAME;
            if (qnn.logCreate) {
                qnn.logCreate(qnnLogCallback, QNN_LOG_LEVEL_ERROR, &logHandle);
            }
            qnn.backendCreate(logHandle, nullptr, &backendHandle);
            if (qnn.deviceCreate) {
                qnn.deviceCreate(nullptr, nullptr, &deviceHandle);
            }
        }
        diagnosticsPerformed = false;
    }
    bool isLoaded() const {
        return backendHandle != nullptr;
    }
    bool hasGraph(int M, int K, int N) {
        std::string key = std::to_string(M) + "_" + std::to_string(K) + "_" + std::to_string(N);
        return graphCache.find(key) != graphCache.end();
    }
    const QnnInterface_t* getInterface() const {
        return backendContainer.interfaceProviderTable;
    }
    Qnn_BackendHandle_t getBackendHandle() const {
        return backendHandle;
    }
    Qnn_DeviceHandle_t getDeviceHandle() const {
        return deviceHandle;
    }
    Qnn_MemHandle_t registerIonMemory(Qnn_ContextHandle_t context, int fd, uint32_t* dims, uint32_t rank, Qnn_DataType_t dataType) {
        if (!qnnMemRegister || !context) return nullptr;
        Qnn_MemDescriptor_t memDesc;
        std::memset(&memDesc, 0, sizeof(memDesc));
        memDesc.memShape.numDim = rank;
        memDesc.memShape.dimSize = dims;
        memDesc.memShape.shapeConfig = nullptr;
        memDesc.dataType = dataType;
        memDesc.memType = QNN_MEM_TYPE_ION;
        memDesc.ionInfo.fd = fd;
        Qnn_MemHandle_t memHandle = nullptr;
        Qnn_ErrorHandle_t status = qnnMemRegister(context, &memDesc, 1, &memHandle);
        if (status != QNN_SUCCESS) {
            return nullptr;
        }
        return memHandle;
    }
    void deregisterMemory(Qnn_MemHandle_t memHandle) {
        if (!qnnMemDeRegister || !memHandle) return;
        qnnMemDeRegister(&memHandle, 1);
    }
    void setPowerMode(int powerMode) {
        if (!isLoaded()) return;
        if (currentBackendChoice != 1 && currentBackendChoice != 7) {
            std::cout << "INFO: Performance mode setting stubbed for backend: " << currentBackendChoice << std::endl;
            return;
        }
        auto& qnn = backendContainer.interfaceProviderTable->QNN_INTERFACE_VER_NAME;
        if (!qnn.deviceGetInfrastructure) return;
        QnnDevice_Infrastructure_t deviceInfra = nullptr;
        Qnn_ErrorHandle_t devErr = qnn.deviceGetInfrastructure(&deviceInfra);
        const auto* htaInfra = (devErr == QNN_SUCCESS && deviceInfra) ? 
            reinterpret_cast<const void*>(deviceInfra) : nullptr;
        if (htaInfra) {
        }
    }
    bool isQuantizedType(Qnn_DataType_t dataType) const {
        return dataType == QNN_DATATYPE_SFIXED_POINT_8 ||
               dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
               dataType == QNN_DATATYPE_SFIXED_POINT_16 ||
               dataType == QNN_DATATYPE_UFIXED_POINT_16 ||
               dataType == QNN_DATATYPE_SFIXED_POINT_32 ||
               dataType == QNN_DATATYPE_UFIXED_POINT_32;
    }
    void performDiagnosticsIfNeeded() {
        if (diagnosticsPerformed || !isLoaded()) {
            return;
        }
        diagnosticsPerformed = true;
        if (std::getenv("NPUBLAS_QUIET") != nullptr && std::string(std::getenv("NPUBLAS_QUIET")) == "1") {
            return;
        }
        std::cout << "\n========================================\n";
        std::cout << "QNN BACKEND TENSOR TYPE SUPPORT DIAGNOSTICS\n";
        std::cout << "========================================\n";
        Qnn_ErrorHandle_t status = validateOpSupport();
        if (status == QNN_SUCCESS) {
            std::cout << "  FullyConnected (SFIXED_POINT_8 -> SFIXED_POINT_16): SUPPORTED\n";
        } else {
            std::cout << "  FullyConnected (SFIXED_POINT_8 -> SFIXED_POINT_16): UNSUPPORTED (Error Code: 0x" << std::hex << status << std::dec << ")\n";
        }
        std::cout << "========================================\n\n";
    }
    Qnn_ErrorHandle_t validateOpSupport() {
        auto& qnn = backendContainer.interfaceProviderTable->QNN_INTERFACE_VER_NAME;
        if (!qnn.backendValidateOpConfig) {
            return 0xFFFFFFFF;
        }
        uint32_t dimsA[2] = {4, 4};
        uint32_t dimsB[2] = {4, 4};
        uint32_t dimsC[2] = {4, 4};
        uint32_t dimsBias[1] = {4};
        std::vector<int32_t> localBias(4, 0);
        Qnn_Tensor_t inputs[3];
        std::memset(inputs, 0, sizeof(inputs));
        inputs[0].version = QNN_TENSOR_VERSION_1;
        inputs[0].v1.id = 0;
        inputs[0].v1.name = "A";
        inputs[0].v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        inputs[0].v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
        inputs[0].v1.rank = 2;
        inputs[0].v1.dimensions = dimsA;
        inputs[0].v1.memType = QNN_TENSORMEMTYPE_RAW;
        inputs[0].v1.clientBuf.data = nullptr;
        inputs[0].v1.clientBuf.dataSize = 0;
        inputs[1].version = QNN_TENSOR_VERSION_1;
        inputs[1].v1.id = 1;
        inputs[1].v1.name = "B";
        inputs[1].v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        inputs[1].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        inputs[1].v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
        inputs[1].v1.rank = 2;
        inputs[1].v1.dimensions = dimsB;
        inputs[1].v1.memType = QNN_TENSORMEMTYPE_RAW;
        inputs[1].v1.clientBuf.data = nullptr;
        inputs[1].v1.clientBuf.dataSize = 0;
        inputs[2].version = QNN_TENSOR_VERSION_1;
        inputs[2].v1.id = 3;
        inputs[2].v1.name = "bias";
        inputs[2].v1.type = QNN_TENSOR_TYPE_STATIC;
        inputs[2].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        inputs[2].v1.dataType = QNN_DATATYPE_SFIXED_POINT_32;
        inputs[2].v1.rank = 1;
        inputs[2].v1.dimensions = dimsBias;
        inputs[2].v1.memType = QNN_TENSORMEMTYPE_RAW;
        inputs[2].v1.clientBuf.data = localBias.data();
        inputs[2].v1.clientBuf.dataSize = 4 * sizeof(int32_t);
        Qnn_Tensor_t outputs[1];
        std::memset(outputs, 0, sizeof(outputs));
        outputs[0].version = QNN_TENSOR_VERSION_1;
        outputs[0].v1.id = 2;
        outputs[0].v1.name = "C";
        outputs[0].v1.type = QNN_TENSOR_TYPE_APP_READ;
        outputs[0].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        outputs[0].v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
        outputs[0].v1.rank = 2;
        outputs[0].v1.dimensions = dimsC;
        outputs[0].v1.memType = QNN_TENSORMEMTYPE_RAW;
        outputs[0].v1.clientBuf.data = nullptr;
        outputs[0].v1.clientBuf.dataSize = 0;
        inputs[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        inputs[0].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset = 127;
        inputs[1].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        inputs[1].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        inputs[1].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        inputs[1].v1.quantizeParams.scaleOffsetEncoding.offset = 127;
        inputs[2].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        inputs[2].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        inputs[2].v1.quantizeParams.scaleOffsetEncoding.offset = 0;
        outputs[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        outputs[0].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        outputs[0].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        outputs[0].v1.quantizeParams.scaleOffsetEncoding.offset = 32768;
        Qnn_OpConfig_t opConfig;
        std::memset(&opConfig, 0, sizeof(opConfig));
        opConfig.version = QNN_OPCONFIG_VERSION_1;
        opConfig.v1.name = "fc";
        opConfig.v1.packageName = "qti.aisw";
        opConfig.v1.typeName = "FullyConnected";
        opConfig.v1.numOfParams = 0;
        opConfig.v1.params = nullptr;
        opConfig.v1.numOfInputs = 3;
        opConfig.v1.inputTensors = inputs;
        opConfig.v1.numOfOutputs = 1;
        opConfig.v1.outputTensors = outputs;
        Qnn_ErrorHandle_t status = qnn.backendValidateOpConfig(backendHandle, opConfig);
        return status;
    }
    QnnCachedGraph& getOrCreateGraph(int M, int K, int N, const uint8_t* staticWeightsPtr = nullptr) {
        auto t_start = std::chrono::high_resolution_clock::now();
        bool useStaticWeights = std::getenv("USE_STATIC_WEIGHTS") != nullptr && std::string(std::getenv("USE_STATIC_WEIGHTS")) == "1";
        std::string key = std::to_string(M) + "_" + std::to_string(K) + "_" + std::to_string(N);
        if (useStaticWeights && staticWeightsPtr) {
            key += "_" + std::to_string(reinterpret_cast<uintptr_t>(staticWeightsPtr));
        }
        auto it = graphCache.find(key);
        if (it != graphCache.end()) {
            return *it->second;
        }
        auto& cachedGraphPtr = graphCache[key];
        cachedGraphPtr = std::make_unique<QnnCachedGraph>();
        QnnCachedGraph& cachedGraph = *cachedGraphPtr;
        cachedGraph.isValid = false;
        auto& qnn = backendContainer.interfaceProviderTable->QNN_INTERFACE_VER_NAME;
        Qnn_ContextHandle_t contextHandle = nullptr;
        qnn.contextCreate(backendHandle, deviceHandle, nullptr, &contextHandle);
        Qnn_GraphHandle_t graphHandle = nullptr;
        qnn.graphCreate(contextHandle, "fc_graph", nullptr, &graphHandle);
        cachedGraph.contextHandle = contextHandle;
        cachedGraph.graphHandle = graphHandle;
        cachedGraph.dimsA[0] = static_cast<uint32_t>(M);
        cachedGraph.dimsA[1] = static_cast<uint32_t>(K);
        cachedGraph.dimsB[0] = static_cast<uint32_t>(N);
        cachedGraph.dimsB[1] = static_cast<uint32_t>(K);
        cachedGraph.dimsC[0] = static_cast<uint32_t>(M);
        cachedGraph.dimsC[1] = static_cast<uint32_t>(N);
        
        std::memset(&cachedGraph.tensorA, 0, sizeof(cachedGraph.tensorA));
        cachedGraph.tensorA.version = QNN_TENSOR_VERSION_1;
        cachedGraph.tensorA.v1.id = 0;
        cachedGraph.tensorA.v1.name = "A";
        cachedGraph.tensorA.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        cachedGraph.tensorA.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        cachedGraph.tensorA.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
        cachedGraph.tensorA.v1.rank = 2;
        cachedGraph.tensorA.v1.dimensions = cachedGraph.dimsA;
        cachedGraph.tensorA.v1.memType = QNN_TENSORMEMTYPE_RAW;
        cachedGraph.tensorA.v1.clientBuf.data = nullptr;
        cachedGraph.tensorA.v1.clientBuf.dataSize = 0;
        
        std::memset(&cachedGraph.tensorB, 0, sizeof(cachedGraph.tensorB));
        cachedGraph.tensorB.version = QNN_TENSOR_VERSION_1;
        cachedGraph.tensorB.v1.id = 1;
        cachedGraph.tensorB.v1.name = "B";
        if (useStaticWeights) {
            cachedGraph.tensorB.v1.type = QNN_TENSOR_TYPE_STATIC;
        } else {
            cachedGraph.tensorB.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        }
        cachedGraph.tensorB.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        cachedGraph.tensorB.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
        cachedGraph.tensorB.v1.rank = 2;
        cachedGraph.tensorB.v1.dimensions = cachedGraph.dimsB;
        cachedGraph.tensorB.v1.memType = QNN_TENSORMEMTYPE_RAW;
        if (useStaticWeights) {
            if (staticWeightsPtr) {
                cachedGraph.staticWeights.assign(staticWeightsPtr, staticWeightsPtr + N * K);
            } else {
                cachedGraph.staticWeights.assign(N * K, 127);
            }
            cachedGraph.tensorB.v1.clientBuf.data = cachedGraph.staticWeights.data();
            cachedGraph.tensorB.v1.clientBuf.dataSize = N * K * sizeof(uint8_t);
        } else {
            cachedGraph.tensorB.v1.clientBuf.data = nullptr;
            cachedGraph.tensorB.v1.clientBuf.dataSize = 0;
        }
        
        std::memset(&cachedGraph.tensorC, 0, sizeof(cachedGraph.tensorC));
        cachedGraph.tensorC.version = QNN_TENSOR_VERSION_1;
        cachedGraph.tensorC.v1.id = 2;
        cachedGraph.tensorC.v1.name = "C";
        cachedGraph.tensorC.v1.type = QNN_TENSOR_TYPE_APP_READ;
        cachedGraph.tensorC.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        cachedGraph.tensorC.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
        cachedGraph.tensorC.v1.rank = 2;
        cachedGraph.tensorC.v1.dimensions = cachedGraph.dimsC;
        cachedGraph.tensorC.v1.memType = QNN_TENSORMEMTYPE_RAW;
        cachedGraph.tensorC.v1.clientBuf.data = nullptr;
        cachedGraph.tensorC.v1.clientBuf.dataSize = 0;
        
        cachedGraph.dimsBias[0] = static_cast<uint32_t>(N);
        std::memset(&cachedGraph.tensorBias, 0, sizeof(cachedGraph.tensorBias));
        cachedGraph.tensorBias.version = QNN_TENSOR_VERSION_1;
        cachedGraph.tensorBias.v1.id = 3;
        cachedGraph.tensorBias.v1.name = "bias";
        cachedGraph.tensorBias.v1.type = QNN_TENSOR_TYPE_STATIC;
        cachedGraph.tensorBias.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        cachedGraph.tensorBias.v1.dataType = QNN_DATATYPE_SFIXED_POINT_32;
        cachedGraph.tensorBias.v1.rank = 1;
        cachedGraph.tensorBias.v1.dimensions = cachedGraph.dimsBias;
        cachedGraph.tensorBias.v1.memType = QNN_TENSORMEMTYPE_RAW;
        cachedGraph.biasData.assign(N, 0);
        cachedGraph.tensorBias.v1.clientBuf.data = cachedGraph.biasData.data();
        cachedGraph.tensorBias.v1.clientBuf.dataSize = N * sizeof(int32_t);
        cachedGraph.tensorA.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        cachedGraph.tensorA.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        cachedGraph.tensorA.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        cachedGraph.tensorA.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
        cachedGraph.tensorB.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        cachedGraph.tensorB.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        cachedGraph.tensorB.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        cachedGraph.tensorB.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
        cachedGraph.tensorC.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        cachedGraph.tensorC.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        cachedGraph.tensorC.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        cachedGraph.tensorC.v1.quantizeParams.scaleOffsetEncoding.offset = 32768;
        cachedGraph.tensorBias.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        cachedGraph.tensorBias.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        cachedGraph.tensorBias.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        cachedGraph.tensorBias.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
        if (qnn.tensorCreateGraphTensor(graphHandle, &cachedGraph.tensorA) != QNN_SUCCESS) {
            std::cerr << "ERROR: Failed to create graph tensor A\n";
        }
        if (qnn.tensorCreateGraphTensor(graphHandle, &cachedGraph.tensorB) != QNN_SUCCESS) {
            std::cerr << "ERROR: Failed to create graph tensor B\n";
        }
        if (qnn.tensorCreateGraphTensor(graphHandle, &cachedGraph.tensorC) != QNN_SUCCESS) {
            std::cerr << "ERROR: Failed to create graph tensor C\n";
        }
        if (qnn.tensorCreateGraphTensor(graphHandle, &cachedGraph.tensorBias) != QNN_SUCCESS) {
            std::cerr << "ERROR: Failed to create graph tensor Bias\n";
        }
        Qnn_OpConfig_t opConfig;
        std::memset(&opConfig, 0, sizeof(opConfig));
        opConfig.version = QNN_OPCONFIG_VERSION_1;
        opConfig.v1.name = "fc_op";
        opConfig.v1.packageName = "qti.aisw";
        opConfig.v1.typeName = "FullyConnected";
        opConfig.v1.numOfParams = 0;
        opConfig.v1.params = nullptr;
        opConfig.v1.numOfInputs = 3;
        Qnn_Tensor_t inputs[3] = { cachedGraph.tensorA, cachedGraph.tensorB, cachedGraph.tensorBias };
        opConfig.v1.inputTensors = inputs;
        opConfig.v1.numOfOutputs = 1;
        opConfig.v1.outputTensors = &cachedGraph.tensorC;
        Qnn_ErrorHandle_t status = qnn.graphAddNode(graphHandle, opConfig);
        if (status != QNN_SUCCESS) {
            std::cerr << "ERROR: graphAddNode failed with error code: 0x" << std::hex << status << std::dec << std::endl;
        }
        status = qnn.graphFinalize(graphHandle, nullptr, nullptr);
        if (status != QNN_SUCCESS) {
            std::cerr << "ERROR: graphFinalize failed with error code: 0x" << std::hex << status << std::dec << std::endl;
            qnn.contextFree(contextHandle, nullptr);
            static QnnCachedGraph dummyInvalid;
            dummyInvalid.isValid = false;
            return dummyInvalid;
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        if (std::getenv("PROF_NPU") != nullptr && std::string(std::getenv("PROF_NPU")) == "1") {
            std::cerr << "[PROF QNN GRAPH CREATION] M=" << M << " K=" << K << " N=" << N << " | Time: " << duration << " us\n";
        }
        cachedGraph.isValid = true;
        return cachedGraph;
    }
    uint32_t getDataTypeSize(Qnn_DataType_t dataType) const {
        if (dataType == QNN_DATATYPE_INT_16 || dataType == QNN_DATATYPE_UINT_16 || dataType == QNN_DATATYPE_SFIXED_POINT_16 || dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_FLOAT_16) {
            return 2;
        }
        if (dataType == QNN_DATATYPE_INT_32 || dataType == QNN_DATATYPE_UINT_32 || dataType == QNN_DATATYPE_SFIXED_POINT_32 || dataType == QNN_DATATYPE_UFIXED_POINT_32 || dataType == QNN_DATATYPE_FLOAT_32) {
            return 4;
        }
        return 1;
    }
    void execute(QnnCachedGraph& graph, const void* inputA, const void* inputB, void* outputC) {
        auto& qnn = backendContainer.interfaceProviderTable->QNN_INTERFACE_VER_NAME;
        bool useStaticWeights = std::getenv("USE_STATIC_WEIGHTS") != nullptr && std::string(std::getenv("USE_STATIC_WEIGHTS")) == "1";
        Qnn_ErrorHandle_t status;
        if (useStaticWeights) {
            QnnCachedGraph tempGraph;
            tempGraph.isValid = false;
            Qnn_ContextHandle_t contextHandle = nullptr;
            qnn.contextCreate(backendHandle, deviceHandle, nullptr, &contextHandle);
            Qnn_GraphHandle_t graphHandle = nullptr;
            qnn.graphCreate(contextHandle, "fc_temp_graph", nullptr, &graphHandle);
            tempGraph.contextHandle = contextHandle;
            tempGraph.graphHandle = graphHandle;
            tempGraph.dimsA[0] = graph.dimsA[0];
            tempGraph.dimsA[1] = graph.dimsA[1];
            tempGraph.dimsB[0] = graph.dimsB[0];
            tempGraph.dimsB[1] = graph.dimsB[1];
            tempGraph.dimsC[0] = graph.dimsC[0];
            tempGraph.dimsC[1] = graph.dimsC[1];
            std::memset(&tempGraph.tensorA, 0, sizeof(tempGraph.tensorA));
            tempGraph.tensorA.version = QNN_TENSOR_VERSION_1;
            tempGraph.tensorA.v1.id = 0;
            tempGraph.tensorA.v1.name = "A";
            tempGraph.tensorA.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
            tempGraph.tensorA.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
            tempGraph.tensorA.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
            tempGraph.tensorA.v1.rank = 2;
            tempGraph.tensorA.v1.dimensions = tempGraph.dimsA;
            tempGraph.tensorA.v1.memType = QNN_TENSORMEMTYPE_RAW;
            std::memset(&tempGraph.tensorB, 0, sizeof(tempGraph.tensorB));
            tempGraph.tensorB.version = QNN_TENSOR_VERSION_1;
            tempGraph.tensorB.v1.id = 1;
            tempGraph.tensorB.v1.name = "B";
            tempGraph.tensorB.v1.type = QNN_TENSOR_TYPE_STATIC;
            tempGraph.tensorB.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
            tempGraph.tensorB.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
            tempGraph.tensorB.v1.rank = 2;
            tempGraph.tensorB.v1.dimensions = tempGraph.dimsB;
            tempGraph.tensorB.v1.memType = QNN_TENSORMEMTYPE_RAW;
            int N = tempGraph.dimsB[0];
            int K = tempGraph.dimsB[1];
            tempGraph.staticWeights.assign(static_cast<const uint8_t*>(inputB), static_cast<const uint8_t*>(inputB) + N * K);
            tempGraph.tensorB.v1.clientBuf.data = tempGraph.staticWeights.data();
            tempGraph.tensorB.v1.clientBuf.dataSize = N * K * sizeof(uint8_t);
            std::memset(&tempGraph.tensorC, 0, sizeof(tempGraph.tensorC));
            tempGraph.tensorC.version = QNN_TENSOR_VERSION_1;
            tempGraph.tensorC.v1.id = 2;
            tempGraph.tensorC.v1.name = "C";
            tempGraph.tensorC.v1.type = QNN_TENSOR_TYPE_APP_READ;
            tempGraph.tensorC.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
            tempGraph.tensorC.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
            tempGraph.tensorC.v1.rank = 2;
            tempGraph.tensorC.v1.dimensions = tempGraph.dimsC;
            tempGraph.tensorC.v1.memType = QNN_TENSORMEMTYPE_RAW;
            tempGraph.dimsBias[0] = static_cast<uint32_t>(N);
            std::memset(&tempGraph.tensorBias, 0, sizeof(tempGraph.tensorBias));
            tempGraph.tensorBias.version = QNN_TENSOR_VERSION_1;
            tempGraph.tensorBias.v1.id = 3;
            tempGraph.tensorBias.v1.name = "bias";
            tempGraph.tensorBias.v1.type = QNN_TENSOR_TYPE_STATIC;
            tempGraph.tensorBias.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
            tempGraph.tensorBias.v1.dataType = QNN_DATATYPE_SFIXED_POINT_32;
            tempGraph.tensorBias.v1.rank = 1;
            tempGraph.tensorBias.v1.dimensions = tempGraph.dimsBias;
            tempGraph.tensorBias.v1.memType = QNN_TENSORMEMTYPE_RAW;
            tempGraph.biasData.assign(N, 0);
            tempGraph.tensorBias.v1.clientBuf.data = tempGraph.biasData.data();
            tempGraph.tensorBias.v1.clientBuf.dataSize = N * sizeof(int32_t);
            tempGraph.tensorA.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tempGraph.tensorA.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tempGraph.tensorA.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
            tempGraph.tensorA.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
            tempGraph.tensorB.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tempGraph.tensorB.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tempGraph.tensorB.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
            tempGraph.tensorB.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
            tempGraph.tensorC.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tempGraph.tensorC.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tempGraph.tensorC.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
            tempGraph.tensorC.v1.quantizeParams.scaleOffsetEncoding.offset = 32768;
            tempGraph.tensorBias.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tempGraph.tensorBias.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tempGraph.tensorBias.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
            tempGraph.tensorBias.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
            qnn.tensorCreateGraphTensor(graphHandle, &tempGraph.tensorA);
            qnn.tensorCreateGraphTensor(graphHandle, &tempGraph.tensorB);
            qnn.tensorCreateGraphTensor(graphHandle, &tempGraph.tensorC);
            qnn.tensorCreateGraphTensor(graphHandle, &tempGraph.tensorBias);
            Qnn_OpConfig_t opConfig;
            std::memset(&opConfig, 0, sizeof(opConfig));
            opConfig.version = QNN_OPCONFIG_VERSION_1;
            opConfig.v1.name = "fc_op";
            opConfig.v1.packageName = "qti.aisw";
            opConfig.v1.typeName = "FullyConnected";
            opConfig.v1.numOfParams = 0;
            opConfig.v1.params = nullptr;
            opConfig.v1.numOfInputs = 3;
            Qnn_Tensor_t inputs[3] = { tempGraph.tensorA, tempGraph.tensorB, tempGraph.tensorBias };
            opConfig.v1.inputTensors = inputs;
            opConfig.v1.numOfOutputs = 1;
            opConfig.v1.outputTensors = &tempGraph.tensorC;
            auto t_g_start = std::chrono::high_resolution_clock::now();
            qnn.graphAddNode(graphHandle, opConfig);
            qnn.graphFinalize(graphHandle, nullptr, nullptr);
            auto t_g_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_g_end - t_g_start).count();
            if (std::getenv("PROF_NPU") != nullptr && std::string(std::getenv("PROF_NPU")) == "1") {
                std::cerr << "[PROF QNN GRAPH CREATION] M=" << tempGraph.dimsA[0] << " K=" << tempGraph.dimsA[1] << " N=" << tempGraph.dimsB[0] << " | Time: " << duration << " us\n";
            }
            tempGraph.tensorA.v1.clientBuf.data = const_cast<void*>(inputA);
            tempGraph.tensorA.v1.clientBuf.dataSize = tempGraph.dimsA[0] * tempGraph.dimsA[1] * getDataTypeSize(tempGraph.tensorA.v1.dataType);
            tempGraph.tensorC.v1.clientBuf.data = outputC;
            tempGraph.tensorC.v1.clientBuf.dataSize = tempGraph.dimsC[0] * tempGraph.dimsC[1] * getDataTypeSize(tempGraph.tensorC.v1.dataType);
            Qnn_Tensor_t execInputs[1] = { tempGraph.tensorA };
            status = qnn.graphExecute(tempGraph.graphHandle, execInputs, 1, &tempGraph.tensorC, 1, nullptr, nullptr);
            if (status != QNN_SUCCESS) {
                std::cerr << "ERROR: graphExecute failed with error code: 0x" << std::hex << status << std::dec << std::endl;
            }
            qnn.contextFree(contextHandle, nullptr);
        } else {
            graph.tensorA.v1.clientBuf.data = const_cast<void*>(inputA);
            graph.tensorA.v1.clientBuf.dataSize = graph.dimsA[0] * graph.dimsA[1] * getDataTypeSize(graph.tensorA.v1.dataType);
            graph.tensorB.v1.clientBuf.data = const_cast<void*>(inputB);
            graph.tensorB.v1.clientBuf.dataSize = graph.dimsB[0] * graph.dimsB[1] * getDataTypeSize(graph.tensorB.v1.dataType);
            graph.tensorC.v1.clientBuf.data = outputC;
            graph.tensorC.v1.clientBuf.dataSize = graph.dimsC[0] * graph.dimsC[1] * getDataTypeSize(graph.tensorC.v1.dataType);
            Qnn_Tensor_t inputs[2] = { graph.tensorA, graph.tensorB };
            status = qnn.graphExecute(graph.graphHandle, inputs, 2, &graph.tensorC, 1, nullptr, nullptr);
            if (status != QNN_SUCCESS) {
                std::cerr << "ERROR: graphExecute failed with error code: 0x" << std::hex << status << std::dec << std::endl;
            }
        }
    }
    void executeWithHandles(QnnCachedGraph& graph, Qnn_MemHandle_t memA, Qnn_MemHandle_t memB, Qnn_MemHandle_t memC) {
        auto& qnn = backendContainer.interfaceProviderTable->QNN_INTERFACE_VER_NAME;
        if (memA) {
            graph.tensorA.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            graph.tensorA.v1.memHandle = memA;
        }
        bool useStaticWeights = std::getenv("USE_STATIC_WEIGHTS") != nullptr && std::string(std::getenv("USE_STATIC_WEIGHTS")) == "1";
        if (!useStaticWeights && memB) {
            graph.tensorB.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            graph.tensorB.v1.memHandle = memB;
        }
        if (memC) {
            graph.tensorC.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            graph.tensorC.v1.memHandle = memC;
        }
        Qnn_ErrorHandle_t status;
        if (useStaticWeights) {
            Qnn_Tensor_t inputs[1] = { graph.tensorA };
            status = qnn.graphExecute(graph.graphHandle, inputs, 1, &graph.tensorC, 1, nullptr, nullptr);
        } else {
            Qnn_Tensor_t inputs[2] = { graph.tensorA, graph.tensorB };
            status = qnn.graphExecute(graph.graphHandle, inputs, 2, &graph.tensorC, 1, nullptr, nullptr);
        }
        if (status != QNN_SUCCESS) {
            std::cerr << "ERROR: graphExecute failed with error code: 0x" << std::hex << status << std::dec << std::endl;
        }
    }
};
inline QnnCachedGraph::~QnnCachedGraph() {}
#endif