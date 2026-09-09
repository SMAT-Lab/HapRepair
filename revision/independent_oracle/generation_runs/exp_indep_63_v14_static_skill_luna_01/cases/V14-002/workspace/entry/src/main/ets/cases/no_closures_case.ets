const dailyTotals: number[] = [14, 18, 22];

function firstTwoDayTotal(totals: number[]): number {
  return totals[0] + totals[1];
}

const twoDayTotal: number = firstTwoDayTotal(dailyTotals);
