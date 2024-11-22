class Settings {
  static config: number[] = [5, 10, 15, 20, 25, 30];
}

function computeValue(index: number): number {
  let total: number = 0;
  for (let i: number = 0; i < 10; i++) {
    // warning
    total += (Settings.config[index] * i);
  }
  return total;
}
