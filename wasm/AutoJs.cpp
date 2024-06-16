
#include <emscripten.h>
#include <iostream>

extern "C" {
void print_array(float *pointer, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; j++) {
      std::cout << pointer[i * cols + j] << " ";
    }
  }
}

EMSCRIPTEN_KEEPALIVE
void fit(float *x_pointer, int x_rows, int x_cols, float *y_pointer, int y_rows,
         int y_cols) {
  print_array(x_pointer, x_rows, x_cols);
  std::cout << std::endl;
  print_array(y_pointer, y_rows, y_cols);
  std::cout << std::endl;
}
}
