// 该规则将检查number是否正确使用。
//
// 根据ArkTS高性能编程实践，建议修改。

let intNum = 1;
// intNum is declared as int. Avoid changing it to float.
intNum = 1.1;
let floatNum = 1.3;
// floatNum is declared as float. Avoid changing it to int.
floatNum = 2;