class Config {
  static limits: number[] = [10, 20, 30, 40];
}

function checkValidity(index: number): boolean {
  for (let value of Config.limits) {
    if (Config.limits[index] > value) {  // 不变值反复访问
      return false;
    }
  }
  return true;
}