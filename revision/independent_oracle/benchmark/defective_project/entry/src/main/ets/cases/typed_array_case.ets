function mergeSamples(): number[] {
  const leftSamples: number[] = new Array(11, 17, 23);
  const rightSamples: number[] = new Array(5, 7, 13);
  const merged: number[] = new Array(3);
  for (let index: number = 0; index < 3; index++) {
    merged[index] = leftSamples[index] + rightSamples[index];
  }
  return merged;
}
