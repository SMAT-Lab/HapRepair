const dailyTotals: number[] = [14, 18, 22];

function firstTwoDayTotal(values: number[]): number {
  return values[0] + values[1];
}

const twoDayTotal: number = firstTwoDayTotal(dailyTotals);
