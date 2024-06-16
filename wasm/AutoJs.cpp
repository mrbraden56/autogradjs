
#include <emscripten.h>
#include <iostream>

extern "C" {
EMSCRIPTEN_KEEPALIVE
void fit(float *inputPointer, int inputSize) {
  std::cout << "Received array:" << std::endl;
  for (int i = 0; i < inputSize; ++i) {
    std::cout << inputPointer[i] << " ";
  }
  std::cout << std::endl;
}
}
