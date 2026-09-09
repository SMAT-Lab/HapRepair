# @performance/hp-performance-no-closures

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1e4eb68ae0cac97f`

建议函数内部变量尽量使用参数传递

### Triggering pattern

```arkts
let arr = [0, 1, 2];
function foo() {
  // arr 尽量通过参数传递
  return arr[0] + arr[1];
}
foo();
```

### Repair pattern

```arkts
let arr = [0, 1, 2];
function foo(array: Array<number>): number {
  // arr 尽量通过参数传递
  return array[0] + array[1];
}
foo(arr);
```

### Rationale

arr 尽量通过参数传递，而不是靠全局变量获取
