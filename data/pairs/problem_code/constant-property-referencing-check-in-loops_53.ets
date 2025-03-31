class Config {
  static settings: number[] = [5, 10, 15, 20, 25, 30];
}

function calculateSum(length: number): number {
  let sum: number = 0;
  let positionValue: number = 1;
  for (let i = 0; i < 5; i++) {
    sum += Config.settings[length - positionValue];
    positionValue++;
  }
  return sum;
}
