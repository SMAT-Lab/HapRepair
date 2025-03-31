class Metrics {
  static data: number[] = [3, 6, 9, 12];
}

function computeSum(factor: number): number {
  let sum: number = 0;
  for (let m = 0; m < Metrics.data.length; m++) {
    sum += Metrics.data[factor] + m; // 每次循环重复访问
  }
  return sum;
}