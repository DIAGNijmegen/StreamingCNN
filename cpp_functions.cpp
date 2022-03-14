#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <sstream>

#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>

at::Tensor backward_weight(
    c10::ArrayRef<long int> weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_fp32) {

  return at::cudnn_convolution_backward_weight(
      weight_size,
      grad_output,
      input,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_fp32);
}
at::Tensor backward_input(
    c10::ArrayRef<long int> input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_fp32) {

  return at::cudnn_convolution_backward_input(
      input_size,
      grad_output,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_fp32);
}

// From pytorch/torch/csrc/Module.cpp
void DLPack_Capsule_Destructor(PyObject* data) {
  HANDLE_TH_ERRORS
    DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
  if (dlMTensor) {
    // the dlMTensor has not been consumed, call deleter ourselves
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
  } else {
    // the dlMTensor has been consumed
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
  END_HANDLE_TH_ERRORS_RET()
}

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &backward_weight, "Conv backward_weight cudnn");
  m.def("backward_input", &backward_input, "Conv backward_input cudnn");
  m.def("to_dlpack_with_device_id", [](const at::Tensor& data, int64_t device_id) {
      DLManagedTensor* dlMTensor = at::toDLPack(data);
      dlMTensor->dl_tensor.ctx.device_id = device_id;
      auto capsule = py::capsule(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
      return capsule;
  }, "Specify device_id in dlpack, for cupy to copy to right GPU");
}
