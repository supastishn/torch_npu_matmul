#include <torch/extension.h>
#include "tflite_manager.h"
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
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_npu, "Custom NPU matmul", py::arg("a"), py::arg("b"), py::arg("backend_choice") = 1);
    m.def("matmul_out", &matmul_npu_out, "Custom NPU matmul in-place", py::arg("a"), py::arg("b"), py::arg("out"), py::arg("backend_choice") = 1);
}