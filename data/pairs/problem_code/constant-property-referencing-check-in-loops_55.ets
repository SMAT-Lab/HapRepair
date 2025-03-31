class Constants {
  static values: number[] = [100, 200, 300, 400, 500];
}

function calculateDifference(start: number): number {
  let difference: number = 0;
  for (let k = 0; k < Constants.values.length; k++) {
    difference += Constants.values[k] - Constants.values[start];
  }
  return difference;
}
