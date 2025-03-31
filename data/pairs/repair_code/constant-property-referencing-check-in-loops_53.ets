class Config {
  static settings: number[] = [5, 10, 15, 20, 25, 30];
}

function calculateSum(length: number): number {
  let sum: number = 0;
  for (let i = 0; i < 5; i++) {
    const index = length - (i + 1);  // 提取到循环外部
    sum += Config.settings[index];
  }
  return sum;
}
