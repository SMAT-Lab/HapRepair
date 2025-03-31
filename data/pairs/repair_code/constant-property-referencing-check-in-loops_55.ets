class Constants {
  static values: number[] = [100, 200, 300, 400, 500];
}

function calculateDifference(start: number): number {
  let difference: number = 0;
  const baseValue = Constants.values[start]; // 提取到循环外
  for (let k = 0; k < Constants.values.length; k++) {
    difference += Constants.values[k] - baseValue;
  }
  return difference;
}
