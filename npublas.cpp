#include <torch/extension.h>
#include "tflite_manager.h"
void matmul_out(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(1);
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    matmul_int8(a_ptr, b_ptr, out_ptr, m, k, n, false, false);
}
torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(1);
    auto out = torch::zeros({m, n}, a.options());
    matmul_out(a, b, out);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Custom NPU matmul");
    m.def("matmul_out", &matmul_out, "Custom NPU matmul in-place");
}