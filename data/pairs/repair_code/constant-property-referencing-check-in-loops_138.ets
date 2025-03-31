class Values {
  static list: number[] = [100, 200, 300, 400];
}

function checkHighValue(ref: number): boolean {
  const refValue = Values.list[ref]; // 提取到循环外
  for (let v of Values.list) {
    if (refValue < v) {
      return false;
    }
  }
  return true;
}