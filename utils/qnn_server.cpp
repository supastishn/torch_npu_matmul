#include "qnn_hta_direct.h"
#include <qualcomm/QNN/QnnBackend.h>
#include <qualcomm/QNN/QnnContext.h>
#include <qualcomm/QNN/QnnGraph.h>
#include <qualcomm/QNN/QnnTensor.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstdlib>
struct DataTypeCheck {
    Qnn_DataType_t type;
    const char* name;
};
const DataTypeCheck dataTypesToCheck[] = {
    { QNN_DATATYPE_INT_8, "QNN_DATATYPE_INT_8" },
    { QNN_DATATYPE_UINT_8, "QNN_DATATYPE_UINT_8" },
    { QNN_DATATYPE_INT_16, "QNN_DATATYPE_INT_16" },
    { QNN_DATATYPE_UINT_16, "QNN_DATATYPE_UINT_16" },
    { QNN_DATATYPE_INT_32, "QNN_DATATYPE_INT_32" },
    { QNN_DATATYPE_UINT_32, "QNN_DATATYPE_UINT_32" },
    { QNN_DATATYPE_FLOAT_16, "QNN_DATATYPE_FLOAT_16" },
    { QNN_DATATYPE_FLOAT_32, "QNN_DATATYPE_FLOAT_32" },
    { QNN_DATATYPE_SFIXED_POINT_8, "QNN_DATATYPE_SFIXED_POINT_8" },
    { QNN_DATATYPE_UFIXED_POINT_8, "QNN_DATATYPE_UFIXED_POINT_8" },
    { QNN_DATATYPE_SFIXED_POINT_16, "QNN_DATATYPE_SFIXED_POINT_16" },
    { QNN_DATATYPE_UFIXED_POINT_16, "QNN_DATATYPE_UFIXED_POINT_16" }
};
bool isQuantizedType(Qnn_DataType_t dataType) {
    return dataType == QNN_DATATYPE_SFIXED_POINT_8 ||
           dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
           dataType == QNN_DATATYPE_SFIXED_POINT_16 ||
           dataType == QNN_DATATYPE_UFIXED_POINT_16 ||
           dataType == QNN_DATATYPE_SFIXED_POINT_32 ||
           dataType == QNN_DATATYPE_UFIXED_POINT_32;
}
Qnn_ErrorHandle_t checkDataTypeSupport(const QnnInterface_t* interfaceProvider, Qnn_BackendHandle_t backendHandle, Qnn_DataType_t dataType, Qnn_DataType_t outputDataType) {
    auto& qnn = interfaceProvider->QNN_INTERFACE_VER_NAME;
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
    inputs[0].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    inputs[0].v1.dataType = dataType;
    inputs[0].v1.rank = 2;
    inputs[0].v1.dimensions = dimsA;
    inputs[1].version = QNN_TENSOR_VERSION_1;
    inputs[1].v1.id = 1;
    inputs[1].v1.name = "B";
    inputs[1].v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    inputs[1].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    inputs[1].v1.dataType = dataType;
    inputs[1].v1.rank = 2;
    inputs[1].v1.dimensions = dimsB;
    inputs[2].version = QNN_TENSOR_VERSION_1;
    inputs[2].v1.id = 3;
    inputs[2].v1.name = "bias";
    inputs[2].v1.type = QNN_TENSOR_TYPE_STATIC;
    inputs[2].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    inputs[2].v1.dataType = QNN_DATATYPE_SFIXED_POINT_32;
    inputs[2].v1.rank = 1;
    inputs[2].v1.dimensions = dimsBias;
    inputs[2].v1.clientBuf.data = localBias.data();
    inputs[2].v1.clientBuf.dataSize = 4 * sizeof(int32_t);
    Qnn_Tensor_t outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].version = QNN_TENSOR_VERSION_1;
    outputs[0].v1.id = 2;
    outputs[0].v1.name = "C";
    outputs[0].v1.type = QNN_TENSOR_TYPE_APP_READ;
    outputs[0].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    outputs[0].v1.dataType = outputDataType;
    outputs[0].v1.rank = 2;
    outputs[0].v1.dimensions = dimsC;
    if (isQuantizedType(dataType)) {
        inputs[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        inputs[0].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset = 0;
        inputs[1].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        inputs[1].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        inputs[1].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        inputs[1].v1.quantizeParams.scaleOffsetEncoding.offset = 0;
        inputs[2].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        inputs[2].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        inputs[2].v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    } else {
        inputs[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        inputs[1].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        inputs[2].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    }
    if (isQuantizedType(outputDataType)) {
        outputs[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        outputs[0].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        outputs[0].v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        outputs[0].v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    } else {
        outputs[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    }
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
void firstRunDiagnosis(const QnnInterface_t* interfaceProvider, const std::string& libraryPath) {
    auto& qnn = interfaceProvider->QNN_INTERFACE_VER_NAME;
    Qnn_BackendHandle_t backendHandle = nullptr;
    if (qnn.backendCreate(nullptr, nullptr, &backendHandle) != QNN_SUCCESS) {
        std::cerr << "Failed to create backend for diagnostics." << std::endl;
        return;
    }
    std::cout << "\n========================================\n";
    std::cout << "QNN BACKEND TENSOR TYPE SUPPORT DIAGNOSTICS\n";
    std::cout << "Library: " << libraryPath << "\n";
    std::cout << "========================================\n";
    for (const auto& dt : dataTypesToCheck) {
        Qnn_ErrorHandle_t status = checkDataTypeSupport(interfaceProvider, backendHandle, dt.type, QNN_DATATYPE_SFIXED_POINT_16);
        if (status == QNN_SUCCESS) {
            std::cout << "  " << dt.name << " (FC to SFIXED_POINT_16): SUPPORTED\n";
        } else {
            std::cout << "  " << dt.name << " (FC to SFIXED_POINT_16): UNSUPPORTED (Error Code: 0x" << std::hex << status << std::dec << ")\n";
        }
    }
    std::cout << "========================================\n\n";
    qnn.backendFree(backendHandle);
}
bool compileMatMulGraph(const QnnInterface_t* interfaceProvider, const std::string& libraryPath, int M, int K, int N, int backendChoice) {
    auto& qnn = interfaceProvider->QNN_INTERFACE_VER_NAME;
    Qnn_BackendHandle_t backendHandle = nullptr;
    if (qnn.backendCreate(nullptr, nullptr, &backendHandle) != QNN_SUCCESS) {
        return false;
    }
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    if (qnn.deviceCreate && qnn.deviceCreate(nullptr, nullptr, &deviceHandle) != QNN_SUCCESS) {
        deviceHandle = nullptr;
    }
    Qnn_ContextHandle_t contextHandle = nullptr;
    if (qnn.contextCreate(backendHandle, deviceHandle, nullptr, &contextHandle) != QNN_SUCCESS) {
        if (deviceHandle && qnn.deviceFree) qnn.deviceFree(deviceHandle);
        qnn.backendFree(backendHandle);
        return false;
    }
    Qnn_GraphHandle_t graphHandle = nullptr;
    if (qnn.graphCreate(contextHandle, "matmul_graph", nullptr, &graphHandle) != QNN_SUCCESS) {
        qnn.contextFree(contextHandle, nullptr);
        if (deviceHandle && qnn.deviceFree) qnn.deviceFree(deviceHandle);
        qnn.backendFree(backendHandle);
        return false;
    }
    uint32_t dimsA[2] = {static_cast<uint32_t>(M), static_cast<uint32_t>(K)};
    uint32_t dimsB[2] = {static_cast<uint32_t>(N), static_cast<uint32_t>(K)};
    uint32_t dimsC[2] = {static_cast<uint32_t>(M), static_cast<uint32_t>(N)};
    uint32_t dimsBias[1] = {static_cast<uint32_t>(N)};
    std::vector<int32_t> localBias(N, 0);
    Qnn_DataType_t dataType = QNN_DATATYPE_UFIXED_POINT_8;
    Qnn_DataType_t outputDataType = QNN_DATATYPE_UFIXED_POINT_16;
    Qnn_Tensor_t tensorA = QNN_TENSOR_INIT;
    tensorA.version = QNN_TENSOR_VERSION_1;
    tensorA.v1.id = 0;
    tensorA.v1.name = "A";
    tensorA.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    tensorA.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    tensorA.v1.dataType = dataType;
    tensorA.v1.rank = 2;
    tensorA.v1.dimensions = dimsA;
    tensorA.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    tensorA.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    tensorA.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
    tensorA.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
    qnn.tensorCreateGraphTensor(graphHandle, &tensorA);
    Qnn_Tensor_t tensorB = QNN_TENSOR_INIT;
    tensorB.version = QNN_TENSOR_VERSION_1;
    tensorB.v1.id = 1;
    tensorB.v1.name = "B";
    tensorB.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    tensorB.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    tensorB.v1.dataType = dataType;
    tensorB.v1.rank = 2;
    tensorB.v1.dimensions = dimsB;
    tensorB.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    tensorB.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    tensorB.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
    tensorB.v1.quantizeParams.scaleOffsetEncoding.offset = 127;
    qnn.tensorCreateGraphTensor(graphHandle, &tensorB);
    Qnn_Tensor_t tensorC = QNN_TENSOR_INIT;
    tensorC.version = QNN_TENSOR_VERSION_1;
    tensorC.v1.id = 2;
    tensorC.v1.name = "C";
    tensorC.v1.type = QNN_TENSOR_TYPE_APP_READ;
    tensorC.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    tensorC.v1.dataType = outputDataType;
    tensorC.v1.rank = 2;
    tensorC.v1.dimensions = dimsC;
    tensorC.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    tensorC.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    tensorC.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
    tensorC.v1.quantizeParams.scaleOffsetEncoding.offset = 32768;
    qnn.tensorCreateGraphTensor(graphHandle, &tensorC);
    Qnn_Tensor_t tensorBias = QNN_TENSOR_INIT;
    tensorBias.version = QNN_TENSOR_VERSION_1;
    tensorBias.v1.id = 3;
    tensorBias.v1.name = "bias";
    tensorBias.v1.type = QNN_TENSOR_TYPE_STATIC;
    tensorBias.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    tensorBias.v1.dataType = QNN_DATATYPE_SFIXED_POINT_32;
    tensorBias.v1.rank = 1;
    tensorBias.v1.dimensions = dimsBias;
    tensorBias.v1.clientBuf.data = localBias.data();
    tensorBias.v1.clientBuf.dataSize = N * sizeof(int32_t);
    tensorBias.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    tensorBias.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    tensorBias.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
    tensorBias.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    qnn.tensorCreateGraphTensor(graphHandle, &tensorBias);
    Qnn_OpConfig_t opConfig;
    std::memset(&opConfig, 0, sizeof(opConfig));
    opConfig.version = QNN_OPCONFIG_VERSION_1;
    opConfig.v1.name = "matmul_op";
    opConfig.v1.packageName = "qti.aisw";
    opConfig.v1.typeName = "FullyConnected";
    opConfig.v1.numOfParams = 0;
    opConfig.v1.params = nullptr;
    opConfig.v1.numOfInputs = 3;
    Qnn_Tensor_t inputs[3] = {tensorA, tensorB, tensorBias};
    opConfig.v1.inputTensors = inputs;
    opConfig.v1.numOfOutputs = 1;
    opConfig.v1.outputTensors = &tensorC;
    if (qnn.graphAddNode(graphHandle, opConfig) != QNN_SUCCESS) {
        qnn.contextFree(contextHandle, nullptr);
        if (deviceHandle && qnn.deviceFree) qnn.deviceFree(deviceHandle);
        qnn.backendFree(backendHandle);
        return false;
    }
    if (qnn.graphFinalize(graphHandle, nullptr, nullptr) != QNN_SUCCESS) {
        qnn.contextFree(contextHandle, nullptr);
        if (deviceHandle && qnn.deviceFree) qnn.deviceFree(deviceHandle);
        qnn.backendFree(backendHandle);
        return false;
    }
    uint64_t binarySize = 0;
    if (qnn.contextGetBinarySize(contextHandle, &binarySize) == QNN_SUCCESS && binarySize > 0) {
        std::vector<uint8_t> buffer(binarySize);
        uint64_t writtenSize = 0;
        if (qnn.contextGetBinary(contextHandle, buffer.data(), binarySize, &writtenSize) == QNN_SUCCESS) {
            std::string outFilename = "matmuls/matmul_" + std::to_string(M) + "x" + std::to_string(K) + "_" + std::to_string(K) + "x" + std::to_string(N) + ".bin";
            std::ofstream outFile(outFilename, std::ios::binary);
            if (outFile) {
                outFile.write(reinterpret_cast<const char*>(buffer.data()), writtenSize);
                std::cout << "Successfully compiled and saved context binary to: " << outFilename << " (" << writtenSize << " bytes)\n";
            }
        }
    }
    qnn.contextFree(contextHandle, nullptr);
    if (deviceHandle && qnn.deviceFree) qnn.deviceFree(deviceHandle);
    qnn.backendFree(backendHandle);
    return true;
}
int main(int argc, char* argv[]) {
    std::string libraryPath = "libs/libQnnHtp.so";
    if (argc > 1) {
        libraryPath = argv[1];
    }
    QnnLoadedBackendContainer loadedBackend = loadQnnBackendSharedLibrary(libraryPath);
    if (!loadedBackend.librarySharedObjectHandle || !loadedBackend.interfaceProviderTable) {
        libraryPath = "libs/libQnnDsp.so";
        loadedBackend = loadQnnBackendSharedLibrary(libraryPath);
    }
    if (!loadedBackend.librarySharedObjectHandle || !loadedBackend.interfaceProviderTable) {
        std::cerr << "CRITICAL ERROR: Failed to load any QNN backend library!" << std::endl;
        return 1;
    }
    firstRunDiagnosis(loadedBackend.interfaceProviderTable, libraryPath);
    system("mkdir -p matmuls");
    int serverFd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd < 0) {
        std::cerr << "Failed to create socket." << std::endl;
        unloadQnnBackendSharedLibrary(loadedBackend);
        return 1;
    }
    int opt = 1;
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(9999);
    if (bind(serverFd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind socket to port 9999." << std::endl;
        close(serverFd);
        unloadQnnBackendSharedLibrary(loadedBackend);
        return 1;
    }
    if (listen(serverFd, 5) < 0) {
        std::cerr << "Failed to listen on socket." << std::endl;
        close(serverFd);
        unloadQnnBackendSharedLibrary(loadedBackend);
        return 1;
    }
    std::cout << "Pure C++ Standalone QNN Compiler Server listening on port 9999..." << std::endl;
    while (true) {
        int clientFd = accept(serverFd, nullptr, nullptr);
        if (clientFd < 0) continue;
        char buffer[1024] = {0};
        int bytesRead = read(clientFd, buffer, sizeof(buffer) - 1);
        if (bytesRead > 0) {
            std::string req(buffer);
            if (req.rfind("generate", 0) == 0) {
                std::stringstream ss(req);
                std::string cmd;
                int M = 0, K = 0, N = 0, backendChoice = 1;
                ss >> cmd >> M >> K >> N >> backendChoice;
                std::cout << "Received compilation request: " << M << "x" << K << "x" << N << " (backend=" << backendChoice << ")\n";
                bool success = compileMatMulGraph(loadedBackend.interfaceProviderTable, libraryPath, M, K, N, backendChoice);
                if (success) {
                    send(clientFd, "OK\n", 3, 0);
                } else {
                    send(clientFd, "ERROR\n", 6, 0);
                }
            }
        }
        close(clientFd);
    }
    close(serverFd);
    unloadQnnBackendSharedLibrary(loadedBackend);
    return 0;
}