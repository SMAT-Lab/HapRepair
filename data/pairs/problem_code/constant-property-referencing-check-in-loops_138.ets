class Values {
  static list: number[] = [100, 200, 300, 400];
}

function checkHighValue(ref: number): boolean {
  for (let v of Values.list) {
    if (Values.list[ref] < v) { // 无效的反复访问
      return false;
    }
  }
  return true;
}