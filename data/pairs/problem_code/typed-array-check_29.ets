const arr1: number[] = [1, 2, 3, 4, 5];
const arr2: number[] = [6, 7, 8, 9, 10];
let sumArray: number[] = new Array(arr1.length);
for (let i = 0; i < arr1.length; i++) {
  sumArray[i] = arr1[i] + arr2[i];
}