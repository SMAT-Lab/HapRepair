let floatArray: number[] = [1.1, 2.2, 3.3, 4.4, 5.5];
let doubleArray: number[] = new Array(floatArray.length);
for (let i = 0; i < floatArray.length; i++) {
  doubleArray[i] = floatArray[i] * 2;
}