class Data {
  static values: number[] = [7, 14, 21, 28];
}

function multiplyValues(multiplier: number): number {
  let product: number = 1;
  const multiplierValue = Data.values[multiplier];  // 提取到循环外
  for (let n of Data.values) {
    product *= multiplierValue + n;
  }
  return product;
}