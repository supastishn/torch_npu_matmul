#include <torch/extension.h>
#include "utils/execution.h"
#include "pipeline.h"
void matmul_npu_out(torch::Tensor a, torch::Tensor b, torch::Tensor out, int backend_choice = 1) {
    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(1);
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    matmul_int8(a_ptr, b_ptr, out_ptr, m, k, n, false, false, backend_choice);
}
torch::Tensor matmul_npu(torch::Tensor a, torch::Tensor b, int backend_choice = 1) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto out = torch::zeros({m, n}, a.options());
    matmul_npu_out(a, b, out, backend_choice);
    return out;
}
void matmul_tiled_npu_out(torch::Tensor a, torch::Tensor b, torch::Tensor out, int m_split, int k_split, int n_split, int backend_choice = 1) {
    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(1);
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    matmul_tiled(a_ptr, b_ptr, out_ptr, m, k, n, m_split, k_split, n_split, backend_choice);
}
torch::Tensor matmul_tiled_npu(torch::Tensor a, torch::Tensor b, int m_split, int k_split, int n_split, int backend_choice = 1) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto out = torch::zeros({m, n}, a.options());
    matmul_tiled_npu_out(a, b, out, m_split, k_split, n_split, backend_choice);
    return out;
}
void set_power_mode_npu(int power_mode) {
    QnnHtaExecutionEngine::getInstance().setPowerMode(power_mode);
}
void set_backend_npu(int backend_choice) {
    QnnHtaExecutionEngine::getInstance().loadBackend(backend_choice);
}
void initialize_npublas(int backend_choice = 1) {
    auto& engine = QnnHtaExecutionEngine::getInstance();
    engine.loadBackend(backend_choice);
    engine.performDiagnosticsIfNeeded();
}
bool has_graph_npu(int m, int k, int n, int input_type, int output_type) {
    auto& engine = QnnHtaExecutionEngine::getInstance();
    return engine.hasGraph(m, k, n);
}
void prepare_matmul_npu(int m, int k, int n, int input_type, int output_type) {
    auto& engine = QnnHtaExecutionEngine::getInstance();
    engine.getOrCreateGraph(m, k, n);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_npu, "Custom NPU matmul", py::arg("a"), py::arg("b"), py::arg("backend_choice") = 1);
    m.def("matmul_out", &matmul_npu_out, "Custom NPU matmul in-place", py::arg("a"), py::arg("b"), py::arg("out"), py::arg("backend_choice") = 1);
    m.def("matmul_tiled", &matmul_tiled_npu, "Custom Tiled NPU matmul with pipelining", py::arg("a"), py::arg("b"), py::arg("m_split"), py::arg("k_split"), py::arg("n_split"), py::arg("backend_choice") = 1);
    m.def("matmul_tiled_out", &matmul_tiled_npu_out, "Custom Tiled NPU matmul in-place with pipelining", py::arg("a"), py::arg("b"), py::arg("out"), py::arg("m_split"), py::arg("k_split"), py::arg("n_split"), py::arg("backend_choice") = 1);
    m.def("set_power_mode", &set_power_mode_npu, "Set power mode (0 to 6)", py::arg("power_mode"));
    m.def("set_backend", &set_backend_npu, "Set active dynamic backend (1=HTA, 2=DSP, 3=HTP, 4=GPU)", py::arg("backend_choice"));
    m.def("prepare_matmul", &prepare_matmul_npu, "Pre-compile and cache graph in RAM", py::arg("m"), py::arg("k"), py::arg("n"), py::arg("input_type") = 8, py::arg("output_type") = 16);
    m.def("initialize", &initialize_npublas, "Initialize and warm up dynamic backend", py::arg("backend_choice") = 1);
    m.def("has_graph", &has_graph_npu, "Check if graph is already compiled and cached in RAM", py::arg("m"), py::arg("k"), py::arg("n"), py::arg("input_type") = 8, py::arg("output_type") = 16);
}