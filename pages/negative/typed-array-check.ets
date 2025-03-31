// 数值数组推荐使用TypedArray。
//
// 根据ArkTS高性能编程实践，建议修改。

const typedArray1: number[] = new Array(1, 2, 3);
const typedArray2: number[] = new Array(4, 5, 6);
let res: number[] = new Array(3);
for (let i = 0; i < 3; i++) {
  res[i] = typedArray1[i] + typedArray2[i];
}