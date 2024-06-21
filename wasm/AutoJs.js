class FFN {
  constructor(layers) {
    this.layers = layers;
    this.factory = require("./AutoJsEngine.js");
  }

  encode_array(x, type, Module) {
    const typedArray = type.from(x.flat());
    const heapPointer = Module._malloc(
      typedArray.length * typedArray.BYTES_PER_ELEMENT,
    );
    Module.HEAPF32.set(typedArray, heapPointer >> 2); // Use HEAPF32 for Float32Array

    return heapPointer;
  }

  pass_data(x, y, epochs, step) {
    this.factory().then((Module) => {
      const x_pointer = this.encode_array(x, Float32Array, Module);
      const y_pointer = this.encode_array(y, Float32Array, Module);
      const layers_pointer = this.encode_array(
        this.layers,
        Float32Array,
        Module,
      );

      Module.ccall(
        "fit", // name of the C++ function
        null, // return type
        [
          "number",
          "number",
          "number",
          "number",
          "number",
          "number",
          "number",
          "number",
          "number",
          "number",
          "number",
        ], // argument types
        [
          x_pointer,
          x.length,
          x[0].length,
          y_pointer,
          y.length,
          y[0].length,
          layers_pointer,
          this.layers.length,
          this.layers[0].length,
          epochs,
          step,
        ], // arguments
      );
      Module._free(x_pointer);
      Module._free(y_pointer);
    });
  }

  fit({ x, y_true, epochs = 10, step = 0.01 }) {
    this.pass_data(x, y_true, epochs, step);
  }
}

if (require.main === module) {
  var x = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ];
  var y_true = [[0], [-1], [1], [0]];
  var layers = [
    [4, 30],
    [30, 4],
  ];
  var nn = new FFN(layers);
  const params = { x, y_true, epochs: 10, step: 0.01 };
  nn.fit(params);
}
