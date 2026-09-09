class PricingTable {
  static readonly multipliers: number[] = [2, 3, 5, 7];
}

function applyMultiplier(values: number[], position: number): number {
  let total: number = 0;
  const multiplier: number = PricingTable.multipliers[position];
  for (let index: number = 0; index < values.length; index++) {
    total += values[index] * multiplier;
  }
  return total;
}
