class Scores {
  static values: number[] = [90, 80, 85, 70, 95];
}

function calculateAverage(bonus: number): number {
  let total: number = 0;
  const bonusValue = Scores.values[bonus];  // 提取到循环外
  for (let j = 0; j < Scores.values.length; j++) {
    total += bonusValue * j;
  }
  return total / Scores.values.length;
}