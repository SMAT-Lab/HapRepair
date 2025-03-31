class Data {
  static values: number[] = [7, 14, 21, 28];
}

function multiplyValues(multiplier: number): number {
  let product: number = 1;
  for (let n of Data.values) {
    product *= Data.values[multiplier] + n;  // 固定值重复使用
  }
  return product;
}