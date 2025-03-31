let floatArray = new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5]);
let doubleArray = new Float32Array(floatArray.length);
for (let i = 0; i < floatArray.length; i++) {
    doubleArray[i] = floatArray[i] * 2;
}