class Scores {
  static values: number[] = [90, 80, 85, 70, 95];
}

function calculateAverage(bonus: number): number {
  let total: number = 0;
  for (let j = 0; j < Scores.values.length; j++) {
    total += Scores.values[bonus] * j;  // 每次循环都访问相同值
  }
  return total / Scores.values.length;
}