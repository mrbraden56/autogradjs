function randomUniform(min, max) {
  return Math.random() * (max - min) + min;
}

class Tensor {
  constructor(value, parents, grad, backward) {
    this.value = value;
    this.parents = parents;
    this.grad = grad;
    this._backward = backward;
  }

  static add(x1, x2) {
    if (!(x1 instanceof Tensor)) {
      x1 = new Tensor(x1, null, 0);
    }
    if (!(x2 instanceof Tensor)) {
      x2 = new Tensor(x2, null, 0);
    }

    var y = new Tensor(x1.value + x2.value, [x1, x2], 0);
    var _backward = function _backward() {
      x1.grad += 1 * y.grad;
      x2.grad += 1 * y.grad;
    };
    y._backward = _backward;
    return y;
  }
  static mul(x1, x2) {
    if (!(x1 instanceof Tensor)) {
      x1 = new Tensor(x1, null, 0);
    }
    if (!(x2 instanceof Tensor)) {
      x2 = new Tensor(x2, null, 0);
    }

    var y = new Tensor(x1.value * x2.value, [x1, x2], 0);
    var _backward = function _backward() {
      x1.grad += x2.value * y.grad;
      x2.grad += x1.value * y.grad;
    };
    y._backward = _backward;
    return y;
  }

  static array(x) {
    for (var i = 0; i < x.length; i++) {
      for (var j = 0; j < x[0].length; j++) {
        x[i][j] = new Tensor(x[i][j], null, 0);
      }
    }
    return x;
  }

  backward() {}
}

class Matrix {
  //NOTE: This class operates on the Tensor class

  zeroes(r, c, val, generator) {
    var matrix = [];
    for (var i = 0; i < r; i++) {
      var row = [];
      for (var j = 0; j < c; j++) {
        if (generator) {
          row.push(val(-1, 1));
        } else {
          row.push(val);
        }
      }

      matrix.push(row);
    }
    return matrix;
  }
  static matmul(a, b) {
    var m = a.length; // Number of rows in A
    var k = a[0].length; // Number of columns in A
    var n = b[0].length; // Number of columns in B

    var ldA = k; // Leading dimension of A (number of columns)
    var ldB = n; // Leading dimension of B (number of columns)
    var ldC = n; // Leading dimension of C (number of columns)

    var rows = m; // Number of rows in C (same as A)
    var columns = n; // Number of columns in C (same as B)

    // Create a zero matrix to store the result
    var c = new Matrix().zeroes(rows, columns, 0, false);

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
  }
  forward(x) {
    return Matrix.matmul(x, this.weights);
  }
}

class FFN {
  constructor() {
    this.layers = this.layers = [
      new Layer(10, 32),
      new Layer(32, 64),
      new Layer(64, 32),
      new Layer(32, 1),
    ];
  }

  forward(x) {
    for (var i = 0; i < this.layers.length; i++) {
      x = this.layers[i].forward(x);
    }
    return x;
  }
}

if (require.main === module) {
  x = [
    [-2, 1, 0, -1, 2, 1, -1, 2, -2, 1],
    [1, -1, 2, -2, 1, 0, 2, -1, 0, -3],
    [0, -2, 1, -1, 2, -1, 0, -1, 1, -3],
    [-1, 2, -1, 1, -2, 0, -1, 2, 0, 1],
    [2, -1, 1, -2, 1, 2, -1, 0, 1, -1],
    [0, 1, -2, 2, -1, 1, -1, 0, 2, -1],
  ]; //NOTE: 6X10

  var x = Tensor.array(x);
  nn = new FFN();

  var epochs = 100;
  for (var i = 0; i < epochs; i++) {
    //nn.zero_grad();
    out = nn.forward(x);
    //loss
    //loss.backward()
    //sgd 0.01
    if (i % 10 === 0) {
      console.log(`Epoch: ${i}`);
    }
  }
  if (i % 10 === 0) {
    console.log(`Epoch: ${i}`);
  }
}
