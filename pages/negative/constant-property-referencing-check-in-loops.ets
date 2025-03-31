// 在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数。
//
// 根据ArkTS高性能编程实践，建议修改。

class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}
function getNum(num: number): number {
  /* Year has (12 * 29 =) 348 days at least */
  let total: number = 348;
  for (let index: number = 0x8000; index > 0x8; index >>= 1) {
    // warning
    total += ((Time.info[num - Time.start] & index) !== 0) ? 1 : 0;
  }
  return total;
}