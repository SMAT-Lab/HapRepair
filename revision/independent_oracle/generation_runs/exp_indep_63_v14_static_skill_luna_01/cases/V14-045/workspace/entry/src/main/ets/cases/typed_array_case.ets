function mergeSamples(): number[] {
  const leftSamples: Int32Array = new Int32Array([11, 17, 23]);
  const rightSamples: Int32Array = new Int32Array([5, 7, 13]);
  const merged: Int32Array = new Int32Array(3);
  for (let index: number = 0; index < 3; index++) {
    merged[index] = leftSamples[index] + rightSamples[index];
  }
  return Array.from(merged);
}
