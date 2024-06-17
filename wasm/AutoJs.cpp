#include <cmath>
#include <emscripten.h>
#include <functional> // for std::function
#include <iostream>
#include <utility> // for std::pair

class Tensor;

class Tensor {
public:
  Tensor(float val, float grad, std::pair<Tensor *, Tensor *> par)
      : value(val), gradient(grad), parents(par) {}

  static Tensor *add(Tensor *x1, Tensor *x2) {
    Tensor *y = new Tensor(x1->value + x2->value, 0, std::make_pair(x1, x2));
    auto backward = [&]() {
      x1->gradient += y->gradient;
      x2->gradient += y->gradient;
    };
    y->_backwards = backward;
    return y;
  }
  static Tensor *sub(Tensor *x1, Tensor *x2) {
    Tensor *y = new Tensor(x1->value - x2->value, 0, std::make_pair(x1, x2));
    auto backward = [&]() {
      x1->gradient += y->gradient;
      x2->gradient += y->gradient;
    };
    y->_backwards = backward;
    return y;
  }

  static Tensor *mul(Tensor *x1, Tensor *x2) {
    Tensor *y = new Tensor(x1->value * x2->value, 0, std::make_pair(x1, x2));
    auto backward = [&]() {
      x1->gradient += x2->value * y->gradient;
      x2->gradient += x1->value * y->gradient;
    };
    y->_backwards = backward;
    return y;
  }

  static Tensor *pow(Tensor *x1, Tensor *exp) {
    Tensor *y =
        new Tensor(powf(x1->value, exp->value), 0, std::make_pair(x1, exp));
    auto backward = [&]() {
      x1->gradient +=
          exp->value * powf(x1->value, exp->value - 1) * y->gradient;
      exp->value +=
          (powf(x1->value, exp->value) * log(x1->value)) * y->gradient;
    };
    y->_backwards = backward;
    return y;
  }

  // static Tensor *tanh(Tensor *x) {
  //   Tensor *y = new Tensor((exp(2 * x->value) - 1) / (exp(2 * x->value) + 1),
  //   0,
  //                          std::make_pair(x, exp));
  //   auto backward = [&]() {};
  //   y->_backwards = backward;
  //   return y;
  // }

protected:
  float value;
  float gradient;
  std::pair<Tensor *, Tensor *> parents;
  std::function<void()> _backwards;
};

void print_array(float *pointer, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; j++) {
      std::cout << pointer[i * cols + j] << " ";
    }
  }
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void fit(float *x_pointer, int x_rows, int x_cols, float *y_pointer, int y_rows,
         int y_cols, int epochs, float step) {
  print_array(x_pointer, x_rows, x_cols);
  std::cout << std::endl;
  print_array(y_pointer, y_rows, y_cols);
  std::cout << std::endl;
  std::cout << epochs << " " << step << "\n";
  std::cout << std::endl;
}
}
