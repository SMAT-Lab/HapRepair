// 避免动态声明function与class，仅适用于js/ts。
//
// 根据ArkTS编程规范，建议修改。

function foo(f: boolean, a: number, b: number): number {
  if (f) {
    function add(c: number, d: number): number {
      return c + d;
    }
    return add(a, b);
  } else {
    function sub(e: number, g: number): number {
      return e - g;
    }
    return sub(a, b);
  }
}