function foo(f: boolean, a: number, b: number): number {
  if (f) {
    return add(a, b);
  } else {
    return sub(a, b);
  }
}
function add(c: number, d: number): number {
  return c + d;
}
function sub(e: number, g: number): number {
  return e - g;
}