
#include <algorithm>
#include <cmath>
#include <emscripten.h>
#include <functional> // for std::function
#include <iostream>
#include <random>
#include <set>
#include <utility> // for std::pair
#include <vector>

class Tensor;

class Tensor {
public:
  Tensor()
      : value(0.0f), gradient(0.0f), parents(std::make_pair(nullptr, nullptr)) {
  }
  Tensor(float val)
      : value(val), gradient(0.0f), parents(std::make_pair(nullptr, nullptr)) {}
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

  bool operator<(const Tensor &other) const { return this < &other; }

  void backward() {
    std::vector<Tensor> topo;
    std::set<Tensor> visited;
    std::function<void(Tensor)> build_topo = [&](Tensor v) {
      if (!visited.count(v)) {
        visited.insert(v);
        if (v.parents.first != nullptr) {
          build_topo(v);
        }
        if (v.parents.second != nullptr) {
          build_topo(v);
        }
        topo.push_back(v);
      }
      build_topo(*this);
      this->gradient = 1;
      std::reverse(topo.begin(), topo.end());
      for (const Tensor &v : topo) {
        if (v._backwards) {
          v._backwards();
        }
      }
    };
  }

  float value;
  float gradient;
  std::pair<Tensor *, Tensor *> parents;
  std::function<void()> _backwards;
};

class Matrix {

public:
  Matrix() {}
  Matrix(int rows, int cols)
      : rows(rows), cols(cols), data(rows * cols),
        shape(std::make_pair(rows, cols)) {}
  int rows, cols;
  std::pair<int, int> shape = std::make_pair(rows, cols);
  std::vector<Tensor> data;

  void print_shape() {
    std::cout << "(" << this->shape.first << ", " << this->shape.second
              << ")\n";
  }

  Tensor &operator[](int index) { return this->data[index]; }

  static Matrix array(std::vector<Tensor> data, int rows, int cols) {
    Matrix mat(rows, cols);
    mat.data = data;
    return mat;
  }

  static double uniform_random(double a, double b) {
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator

    std::uniform_real_distribution<> dis(
        a, b);       // Define the distribution for doubles between [a, b)
    return dis(gen); // Generate a random number
  }

  static Matrix zeroes(int rows, int cols) {
    Matrix mat(rows, cols);
    // std::vector<Tensor> arr(rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; j++) {
        mat.data[i * cols + j] = Tensor(0, 0, std::make_pair(nullptr, nullptr));
      }
    }
    return mat;
  }
  static Matrix initialize(int rows, int cols) {
    Matrix mat(rows, cols);
    // std::vector<Tensor> arr(rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; j++) {
        double rand_num = uniform_random(-1.0, 1.0);
        mat.data[i * cols + j] =
            Tensor(rand_num, 0, std::make_pair(nullptr, nullptr));
      }
    }
    return mat;
  }
  //(AB) * (BC) = AC
  static Matrix matmul(Matrix x, Matrix y) {
    Matrix mat(x.shape.first, y.shape.second);
    int xr = x.shape.first;
    int xc = x.shape.second;
    int yr = y.shape.first;
    int yc = y.shape.second;
    // std::vector<Tensor> arr(xr * yc);
    for (int i = 0; i < xr; ++i) {
      for (int j = 0; j < yc; ++j) {
        Tensor sum = Tensor(0, 0, std::make_pair(nullptr, nullptr));
        for (int k = 0; k < xc; ++k) {
          sum = Tensor::add(sum, Tensor::mul(x[i * xc + k], y[k * yc + j]));
        }
        mat.data[i * yc + j] = sum;
      }
    }
    return mat;
  }

  static Matrix add(Matrix x, Matrix y) {

    // std::vector<Tensor> arr(x.size());
    Matrix mat(x.shape.first, x.shape.second);
    for (int i = 0; i < x.shape.first; i++) {
      for (int j = 0; j < x.shape.second; j++) {
        mat.data[i] = Tensor::add(x[i * x.shape.second + j], y[j]);
      }
    }

    return mat;
  }
  static Matrix tanh(Matrix x) {
    // std::vector<Tensor> arr(x.size());
    Matrix mat(x.shape.first, x.shape.second);
    for (int i = 0; i < x.data.size(); i++) {
      mat.data[i] = Tensor::tanh(x[i]);
    }
    return mat;
  }

  static Tensor mse(Matrix predicted, Matrix actual) {
    Tensor mse = Tensor(0);
    int total_elements = predicted.shape.first * predicted.shape.second;
    Tensor pow = Tensor(2);

    for (int i = 0; i < total_elements; ++i) {
      Tensor diff = Tensor::sub(predicted.data[i], actual.data[i]);
      Tensor squared_diff = Tensor::pow(diff, pow);
      mse = Tensor::add(mse, squared_diff);
    }

    return mse;
  }
};

class Layer {

public:
  Layer(int inputs, int outputs) : nin(inputs), nout(outputs) {}
  Layer() : nin(0), nout(0) {}

  Matrix forward(Matrix x) {
    return Matrix::tanh(
        Matrix::add(Matrix::matmul(x, this->weights), this->bias));
  }

  void shape() {}

  int nin;
  int nout;
  Matrix weights = Matrix::initialize(nin, nout);
  Matrix bias = Matrix::initialize(1, nout);
};

class FFN {
public:
  std::vector<Layer> layers; // = {Layer(4, 30), Layer(30, 4)};

  void initializeLayers(float *layers_pointer, int layers_rows,
                        int layers_cols) {
    for (int i = 0; i < layers_rows; ++i) {
      int nin = layers_pointer[i * layers_cols];
      int nout = layers_pointer[i * layers_cols + 1];
      layers.push_back(Layer(nin, nout));
    }
  }

  Matrix forward(Matrix input) {
    for (int i = 0; i < this->layers.size(); i++) {
      input = this->layers[i].forward(input);
      std::cout << "Forward done!\n";
    }
    return input;
  }
};

void print_array(float *pointer, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; j++) {
      std::cout << pointer[i * cols + j] << " ";
    }
  }
  std::cout << "\n";
}

void test(Matrix x, Matrix y) {
  Matrix matmul = Matrix::matmul(x, y);
  Matrix bias = Matrix::initialize(1, 30);
  Matrix added = Matrix::add(matmul, bias);
  Matrix::tanh(added);
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void fit(float *x_pointer, int x_rows, int x_cols, float *y_pointer, int y_rows,
         int y_cols, float *layers_pointer, int layers_rows, int layers_cols,
         int epochs, float step) {
  Matrix x =
      Matrix::array(Tensor::array(x_pointer, x_rows, x_cols), x_rows, x_cols);
  Matrix y =
      Matrix::array(Tensor::array(y_pointer, y_rows, y_cols), y_rows, y_cols);
  // NOTE: Tests
  test(x, y);

  FFN ffn;
  ffn.initializeLayers(layers_pointer, layers_rows, layers_cols);
  Matrix out = ffn.forward(x);
  out.print_shape();
  Tensor loss = Matrix::mse(out, y);
  std::cout << "Loss: " << loss.value << "\n";
  std::cout << "Grad: " << loss.gradient << "\n";
  std::cout << "Backwards!\n";
  loss.backward();
  std::cout << "Loss: " << loss.value << "\n";
  std::cout << "Grad: " << loss.gradient << "\n";
}
}
