function randomUniform(min, max) {
  return Math.random() * (max - min) + min;
}

class Tensor {
  constructor(value, parents = [], grad = 0, backward = null) {
    this.value = value;
    this.parents = parents;
    this.grad = grad;
    this._backward = backward;
  }

  static add(x1, x2) {
    if (!(x1 instanceof Tensor)) {
      x1 = new Tensor(x1);
    }
    if (!(x2 instanceof Tensor)) {
      x2 = new Tensor(x2);
    }

    var y = new Tensor(x1.value + x2.value, [x1, x2]);
    var _backward = function _backward() {
      x1.grad += 1 * y.grad;
      x2.grad += 1 * y.grad;
    };
    y._backward = _backward;
    return y;
  }

  static sub(x1, x2) {
    if (!(x1 instanceof Tensor)) {
      x1 = new Tensor(x1);
    }
    if (!(x2 instanceof Tensor)) {
      x2 = new Tensor(x2);
    }

    var y = new Tensor(x1.value - x2.value, [x1, x2]);
    var _backward = function _backward() {
      x1.grad += 1 * y.grad;
      x2.grad += 1 * y.grad;
    };
    y._backward = _backward;
    return y;
  }

  static mul(x1, x2) {
    if (!(x1 instanceof Tensor)) {
      x1 = new Tensor(x1);
    }
    if (!(x2 instanceof Tensor)) {
      x2 = new Tensor(x2);
    }

    var y = new Tensor(x1.value * x2.value, [x1, x2]);
    var _backward = function _backward() {
      x1.grad += x2.value * y.grad;
      x2.grad += x1.value * y.grad;
    };
    y._backward = _backward;
    return y;
  }

  static pow(x1, power) {
    if (!(x1 instanceof Tensor)) {
      x1 = new Tensor(x1);
    }

    var y = new Tensor(x1.value ** power, [x1]);
    var _backward = function _backward() {
      x1.grad += power * x1.value ** (power - 1) * y.grad;
    };
    y._backward = _backward;
    return y;
  }

  static relu(x) {
    var val = x.value;
    if (val < 0) {
      val = 0;
    }
    var y = new Tensor(val, [x]);
    var _backward = function _backward() {
      if (y.value > 0) {
        x.grad += y.grad;
      } else {
        x.grad += 0; // This is redundant but makes it clear
      }
    };
    y._backward = _backward;
    return y;
  }

  static tanh(x) {
    var t = (Math.exp(2 * x.value) - 1) / (Math.exp(2 * x.value) + 1);
    var y = new Tensor(t, [x]);
    var _backward = function _backward() {
      x.grad += (1 - t ** 2) * y.grad;
    };
    y._backward = _backward;
    return y;
  }

  static array(x) {
    for (var i = 0; i < x.length; i++) {
      for (var j = 0; j < x[0].length; j++) {
        x[i][j] = new Tensor(x[i][j]);
      }
    }
    return x;
  }

  backward() {
    var topo = [];
    var visited = new Set();

    function build_topo(v) {
      if (!visited.has(v)) {
        visited.add(v);
        if (v.parents) {
          for (const child of v.parents) {
            build_topo(child);
          }
        }
        topo.push(v);
      }
    }

    build_topo(this);
    this.grad = 1;
    topo.reverse();

    for (const v of topo) {
      if (v._backward) {
        v._backward();
      }
    }
  }
}

class Matrix {
  zeroes(r, c, val, generator) {
    var matrix = [];
    for (var i = 0; i < r; i++) {
      var row = [];
      for (var j = 0; j < c; j++) {
        if (generator) {
          var ten_val = new Tensor(val(-1, 1));
          row.push(ten_val);
        } else {
          var ten_val = new Tensor(val);
          row.push(ten_val);
        }
      }
      matrix.push(row);
    }
    return matrix;
  }

  static matmul(a, b) {
    var m = a.length;
    var k = a[0].length;
    var n = b[0].length;

    var c = new Matrix().zeroes(m, n, 0, false);

    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        for (var p = 0; p < k; p++) {
          var product = Tensor.mul(a[i][p], b[p][j]);
          c[i][j] = Tensor.add(c[i][j], product);
        }
      }
    }
    return c;
  }
}

class Layer {
  constructor(nin, nout) {
    this.weights = new Matrix().zeroes(nin, nout, randomUniform, true);
    this.bias = new Matrix().zeroes(1, nout, randomUniform, true);
  }

  forward(x) {
    var matmul = Matrix.matmul(x, this.weights);
    for (var i = 0; i < matmul.length; i++) {
      for (var j = 0; j < matmul[0].length; j++) {
        matmul[i][j] = Tensor.add(matmul[i][j], this.bias[0][j]);
        matmul[i][j] = Tensor.tanh(matmul[i][j]);
      }
    }
    return matmul;
  }
}

class FFN {
  constructor() {
    this.layers_1 = [
      //2x3
      new Layer(3, 32),
      new Layer(32, 64),
      new Layer(64, 256),
      new Layer(256, 64),
      new Layer(64, 1),
    ];

    this.layers = this.layers_1;
  }

  forward(x) {
    for (var i = 0; i < this.layers.length; i++) {
      x = this.layers[i].forward(x);
    }

    return x;
  }

  zero_grad() {
    for (var i = 0; i < this.layers.length; i++) {
      var layer = this.layers[i];
      var r = layer.weights.length;
      var c = layer.weights[0].length;
      for (var j = 0; j < r; j++) {
        for (var k = 0; k < c; k++) {
          layer.weights[j][k].grad = 0;
        }
      }
      for (var j = 0; j < layer.bias.length; j++) {
        layer.bias[j].grad = 0;
      }
    }
  }

  sgd(step) {
    for (var i = 0; i < this.layers.length; i++) {
      var layer = this.layers[i];
      var r = layer.weights.length;
      var c = layer.weights[0].length;
      for (var j = 0; j < r; j++) {
        for (var k = 0; k < c; k++) {
          layer.weights[j][k].value += -layer.weights[j][k].grad * step;
        }
      }
      for (var j = 0; j < layer.bias.length; j++) {
        layer.bias[j].value += -layer.bias[j].grad * step;
      }
    }
  }
}

if (require.main === module) {
  runTests();

  let startTime = performance.now();

  var x = [
    [2, 3, -1],
    [-1, 0, -2],
    [3, 2, 3],
  ]; //3x3
  x = Tensor.array(x);

  var ys = [[0], [-1], [1]];
  ys = Tensor.array(ys);
  var nn = new FFN();

  var epochs = 100;
  for (var epoch = 0; epoch < epochs; epoch++) {
    nn.zero_grad();
    var out = nn.forward(x);

    var loss = new Tensor(0);

    for (var i = 0; i < out.length; i++) {
      for (var j = 0; j < out[0].length; j++) {
        var difference = Tensor.sub(out[i][j], ys[i][j]);
        var mse = Tensor.pow(difference, 2);
        loss = Tensor.add(loss, mse);
      }
    }
    loss.backward();
    nn.sgd(0.01);
    if (epoch % 10 === 0) {
      console.log(`Epoch: ${epoch}, Loss: ${loss.value}`);
    }
  }
  console.log(`Epoch: ${epochs}, Loss: ${loss.value}`);
  console.log(
    `Actual Values" ${ys[0][0].value}, ${ys[1][0].value}, ${ys[2][0].value}`,
  );
  console.log(
    `Pred Values" ${out[0][0].value}, ${out[1][0].value}, ${out[2][0].value}`,
  );
  let endTime = performance.now();
  let executionTime = (endTime - startTime) / 1000;
  console.log(`Execution time: ${executionTime} seconds`);
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(message || `Expected ${actual} to equal ${expected}`);
  }
}

function assertClose(actual, expected, tolerance = 1e-6, message) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(
      message ||
        `Expected ${actual} to be close to ${expected} within ${tolerance}`,
    );
  }
}

function testCreateTensor() {
  let t = new Tensor(5);
  assertEqual(t.value, 5, "Tensor value should be 5");
  assertEqual(t.grad, 0, "Tensor gradient should be 0");
}

function testAddTensors() {
  let t1 = new Tensor(3);
  let t2 = new Tensor(4);
  let t3 = Tensor.add(t1, t2);
  assertEqual(t3.value, 7, "3 + 4 should equal 7");

  t3.grad = 1;
  t3.backward();
  assertEqual(t1.grad, 1, "Gradient of t1 should be 1");
  assertEqual(t2.grad, 1, "Gradient of t2 should be 1");
}

function testSubTensors() {
  let t1 = new Tensor(7);
  let t2 = new Tensor(4);
  let t3 = Tensor.sub(t1, t2);
  assertEqual(t3.value, 3, "7 - 4 should equal 3");

  t3.grad = 1;
  t3.backward();
  assertEqual(t1.grad, 1, "Gradient of t1 should be 1");
  assertEqual(t2.grad, 1, "Gradient of t2 should be 1");
}

function testMulTensors() {
  let t1 = new Tensor(3);
  let t2 = new Tensor(4);
  let t3 = Tensor.mul(t1, t2);
  assertEqual(t3.value, 12, "3 * 4 should equal 12");

  t3.grad = 1;
  t3.backward();
  assertEqual(t1.grad, 4, "Gradient of t1 should be 4");
  assertEqual(t2.grad, 3, "Gradient of t2 should be 3");
}

function testPowTensors() {
  let t1 = new Tensor(2);
  let t2 = Tensor.pow(t1, 3);
  assertEqual(t2.value, 8, "2^3 should equal 8");

  t2.grad = 1;
  t2.backward();
  assertEqual(t1.grad, 12, "Gradient of t1 should be 12");
}

function testReLUTensor() {
  let t1 = new Tensor(-3);
  let t2 = Tensor.relu(t1);
  assertEqual(t2.value, 0, "ReLU of -3 should be 0");

  t2.grad = 1;
  t2.backward();
  assertEqual(t1.grad, 0, "Gradient of t1 should be 0");

  let t3 = new Tensor(3);
  let t4 = Tensor.relu(t3);
  assertEqual(t4.value, 3, "ReLU of 3 should be 3");

  t4.grad = 1;
  t4.backward();
  assertEqual(t3.grad, 1, "Gradient of t3 should be 1");
}

function testMatrixMultiplication() {
  let a = [[new Tensor(1), new Tensor(2)]];
  let b = [[new Tensor(2)], [new Tensor(1)]];
  let c = Matrix.matmul(a, b);
  assertEqual(c[0][0].value, 4, "Matrix multiplication result should be 4");

  c[0][0].grad = 1;
  c[0][0].backward();
  assertEqual(a[0][0].grad, 2, "Gradient of a[0][0] should be 2");
  assertEqual(a[0][1].grad, 1, "Gradient of a[0][1] should be 1");
  assertEqual(b[0][0].grad, 1, "Gradient of b[0][0] should be 1");
  assertEqual(b[1][0].grad, 2, "Gradient of b[1][0] should be 2");
}

function testMse() {
  let y = [[new Tensor(1), new Tensor(1)]];
  let pred = [[new Tensor(2), new Tensor(30)]];
  var loss = new Tensor(0);

  for (var i = 0; i < y.length; i++) {
    for (var j = 0; j < y[0].length; j++) {
      var difference = Tensor.sub(pred[i][j], y[i][j]);
      var mse = Tensor.pow(difference, 2);
      loss = Tensor.add(loss, mse);
    }
  }
  //NOTE: (2-1)**2 + (30-1)**2
  assertEqual(loss.value, 842, "MSE Failed");
}

function runTests() {
  try {
    testCreateTensor();
    testAddTensors();
    testSubTensors();
    testMulTensors();
    testPowTensors();
    testReLUTensor();
    testMatrixMultiplication();
    testMse();
    console.log("All tests passed!");
  } catch (error) {
    console.error("Test failed:", error.message);
  }
}
