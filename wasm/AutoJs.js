class FFN {
  constructor(layers) {
    this.layers = layers;
    this.factory = require("./AutoJsEngine.js");
  }

  encode_array(x, type, Module) {
    const typedArray = type.from(x);
    const heapPointer = Module._malloc(
      typedArray.length * typedArray.BYTES_PER_ELEMENT,
    );
    Module.HEAPF32.set(typedArray, heapPointer >> 2); // Use HEAPF32 for Float32Array

    return heapPointer;
  }

  pass_data(x) {
    this.factory().then((Module) => {
      /* const pointer = this.encode_array(x, Float32Array, Module); */
      const typedArray = Float32Array.from(x);
      const heapPointer = Module._malloc(
        typedArray.length * typedArray.BYTES_PER_ELEMENT,
      );
      Module.HEAPF32.set(typedArray, heapPointer >> 2);
      var pointer = heapPointer;
      Module.ccall(
        "fit", // name of the C++ function
        null, // return type
        ["number", "number"], // argument types
        [pointer, x.length], // arguments
      );
      Module._free(pointer);
    });
  }

  fit({ x, epochs = 10, step = 0.01 }) {
    for (var epoch = 0; epoch < epochs; epoch++) {
      this.pass_data(x);
    }
  }
}

if (require.main === module) {
  var x = [2, 3, -1];
  var ys = [[0], [-1], [1]];
  var layers = [
    [3, 30],
    [30, 1],
  ];
  var nn = new FFN(layers);
  const params = { x, epochs: 1, step: 0.01 };
  nn.fit(params);
}
