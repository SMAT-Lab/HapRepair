// 建议函数内部变量尽量使用参数传递。
//
// 根据ArkTS编程规范，建议修改。

let arr = [0, 1, 2];
function foo() {
  // arr 尽量通过参数传递
  return arr[0] + arr[1];
}
foo();