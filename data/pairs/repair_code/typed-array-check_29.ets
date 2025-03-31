const arr1 = new Int8Array([1, 2, 3, 4, 5]);
const arr2 = new Int8Array([6, 7, 8, 9, 10]);
let sumArray = new Int8Array(arr1.length);
for (let i = 0; i < arr1.length; i++) {
    sumArray[i] = arr1[i] + arr2[i];
}