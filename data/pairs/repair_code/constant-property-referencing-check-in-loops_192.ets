class Config {
  static limits: number[] = [10, 20, 30, 40];
}

function checkValidity(index: number): boolean {
  const limitThreshold = Config.limits[index];  // 提取到循环外
  for (let value of Config.limits) {
    if (limitThreshold > value) {
      return false;
    }
  }
  return true;
}