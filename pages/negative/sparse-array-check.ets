// 建议避免使用稀疏数组。
//
// 根据ArkTS高性能编程实践，建议修改。

let count = 100000;
let result: number[] = new Array(count);
result = new Array();
result[9999] = 0;