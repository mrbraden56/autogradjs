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
      x1.grad += power * y.value ** (power - 1) * y.grad;
    };
    y._backward = _backward;
    return y;
  }

  static relu(x) {
    var val = x.value;
    if (val < 0) {
      val = 0;
    }
    var y = new Tensor(val);
    var _backward = function _backward() {
      this.grad += (y.value > 0) * y.grad;
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
        // Apply ReLU activation function
        c[i][j] = Tensor.relu(c[i][j]);
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
    this.layers = [
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

  zero_grad() {
    for (var i = 0; i < this.layers.length; i++) {
      var r = this.layers[i].weights.length;
      var c = this.layers[i].weights[0].length;
      for (var j = 0; j < r; j++) {
        for (var k = 0; k < c; k++) {
          this.layers[i].weights[j][k].grad = 0;
        }
      }
    }
  }

  sgd(step) {
    for (var i = 0; i < this.layers.length; i++) {
      var r = this.layers[i].weights.length;
      var c = this.layers[i].weights[0].length;
      for (var j = 0; j < r; j++) {
        for (var k = 0; k < c; k++) {
          console.log("here")
          this.layers[i].weights[j][k].value +=
            -this.layers[i].weights[j][k].grad * step;
          console.log("here")
        }
      }
    }
  }
}

if (require.main === module) {
  var x = [
    [-2, 1, 0, -1, 2, 1, -1, 2, -2, 1],
    [1, -1, 2, -2, 1, 0, 2, -1, 0, -3],
    [0, -2, 1, -1, 2, -1, 0, -1, 1, -3],
    [-1, 2, -1, 1, -2, 0, -1, 2, 0, 1],
    [2, -1, 1, -2, 1, 2, -1, 0, 1, -1],
    [0, 1, -2, 2, -1, 1, -1, 0, 2, -1],
  ];
  x = Tensor.array(x);

  var ys = [[1, -2, 2, -1, 1, -2]];
  ys = Tensor.array(ys);
  var nn = new FFN();

  var epochs = 100;
  for (var i = 0; i < epochs; i++) {
    nn.zero_grad(0.01);
    var out = nn.forward(x);
    var loss = new Tensor(0);

    for (var j = 0; j < out.length; j++) {
      var difference = Tensor.sub(out[j][0], ys[0][j]);
      var mse = Tensor.pow(difference, 2);
      loss = Tensor.add(loss, mse);
    }
    loss.backward();
    nn.sgd(0.01);
    if (i % 1 === 0) {
      console.log(`Epoch: ${i}, Loss: ${loss.value}`);
    }
  }
  console.log(`Epoch: ${epochs}, Loss: ${loss.value}`);
}

//NOTE: TODO
//I think the loss is staying the same because backward() isnt updating the gradients
//Still missing Bias
