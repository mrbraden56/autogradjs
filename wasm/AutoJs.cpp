
#include <cmath>
#include <emscripten.h>
#include <functional> // for std::function
#include <iostream>
#include <random>
#include <utility> // for std::pair
#include <vector>

class Tensor;

class Tensor {
public:
  Tensor()
      : value(0.0f), gradient(0.0f), parents(std::make_pair(nullptr, nullptr)) {
  }
  Tensor(float val, float grad, std::pair<Tensor *, Tensor *> par)
      : value(val), gradient(grad), parents(par) {}

  static Tensor add(Tensor x1, Tensor x2) {
    Tensor y(x1.value + x2.value, 0, std::make_pair(&x1, &x2));
    auto backward = [&]() {
      x1.gradient += y.gradient;
      x2.gradient += y.gradient;
    };
    y._backwards = backward;
    return y;
  }
  static Tensor sub(Tensor x1, Tensor x2) {
    Tensor y(x1.value - x2.value, 0, std::make_pair(&x1, &x2));
    auto backward = [&]() {
      x1.gradient += y.gradient;
      x2.gradient += y.gradient;
    };
    y._backwards = backward;
    return y;
  }

  static Tensor mul(Tensor x1, Tensor x2) {
    Tensor y(x1.value * x2.value, 0, std::make_pair(&x1, &x2));
    auto backward = [&]() {
      x1.gradient += x2.value * y.gradient;
      x2.gradient += x1.value * y.gradient;
    };
    y._backwards = backward;
    return y;
  }

  static Tensor pow(Tensor x1, Tensor exp) {
    Tensor y(powf(x1.value, exp.value), 0, std::make_pair(&x1, &exp));
    auto backward = [&]() {
      x1.gradient += exp.value * powf(x1.value, exp.value - 1) * y.gradient;
      exp.value += (powf(x1.value, exp.value) * log(x1.value)) * y.gradient;
    };
    y._backwards = backward;
    return y;
  }

  static Tensor tanh(Tensor x) {
    float t = (exp(2 * x.value) - 1) / (exp(2 * x.value) + 1);
    Tensor y(t, 0, std::make_pair(&x, nullptr));
    auto backward = [&]() { x.gradient += (1 - powf(t, 2)) * y.gradient; };
    y._backwards = backward;
    return y;
  }

  static std::vector<Tensor> array(float *x, int rows, int cols) {
    std::vector<Tensor> arr(rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; j++) {
        arr[i * cols + j] =
            Tensor(x[i * cols + j], 0, std::make_pair(nullptr, nullptr));
      }
    }
    return arr;
  }

  float value;
  float gradient;
  std::pair<Tensor *, Tensor *> parents;
  std::function<void()> _backwards;
};

class Matrix {

public:
  Matrix() {}

  static double uniform_random(double a, double b) {
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator

    std::uniform_real_distribution<> dis(
        a, b);       // Define the distribution for doubles between [a, b)
    return dis(gen); // Generate a random number
  }

  static std::vector<Tensor> zeroes(int rows, int cols) {
    std::vector<Tensor> arr(rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; j++) {
        arr[i * cols + j] = Tensor(0, 0, std::make_pair(nullptr, nullptr));
      }
    }
    return arr;
  }
  static std::vector<Tensor> initialize(int rows, int cols) {
    std::vector<Tensor> arr(rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; j++) {
        double rand_num = uniform_random(-1.0, 1.0);
        arr[i * cols + j] =
            Tensor(rand_num, 0, std::make_pair(nullptr, nullptr));
      }
    }
    return arr;
  }
  //(AB) * (BC) = AC
  static std::vector<Tensor> matmul(std::vector<Tensor> x, int xr, int xc,
                                    std::vector<Tensor> y, int yr, int yc) {

    std::vector<Tensor> arr(xr * yc);
    for (int i = 0; i < xr; ++i) {
      for (int j = 0; j < yc; ++j) {
        Tensor sum = Tensor(0, 0, std::make_pair(nullptr, nullptr));
        for (int k = 0; k < xc; ++k) {
          sum = Tensor::add(sum, Tensor::mul(x[i * xc + k], y[k * yc + j]));
        }
        arr[i * yc + j] = sum;
      }
    }
    return arr;
  }
  static std::vector<Tensor> tanh() {}
};

class Layer {

public:
  Layer() : nin(0), nout(0) {}

  // Matrix forward() {
  //   Matrix.tanh(Matrix.add(Matrix.matmul(x, this->weights), this->bias))
  // }

  int nin;
  int nout;
  std::vector<Tensor> weights = Matrix::initialize(nin, nout);
  std::vector<Tensor> bias = Matrix::initialize(1, nout);
};

// class FFN {};

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
         int y_cols, float *layers_pointer, int layers_rows, int layers_cols,
         int epochs, float step) {
  std::vector<Tensor> x = Tensor::array(x_pointer, x_rows, x_cols);
  std::vector<Tensor> y = Tensor::array(y_pointer, y_rows, y_cols);
}
}
