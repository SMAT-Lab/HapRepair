class Metrics {
  static data: number[] = [3, 6, 9, 12];
}

function computeSum(factor: number): number {
  let sum: number = 0;
  const factorValue = Metrics.data[factor]; // 提取到循环外
  for (let m = 0; m < Metrics.data.length; m++) {
    sum += factorValue + m;
  }
  return sum;
}