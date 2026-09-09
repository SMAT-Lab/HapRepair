function mergeSamples(): Int16Array {
  const leftSamples: Int16Array = new Int16Array([11, 17, 23]);
  const rightSamples: Int16Array = new Int16Array([5, 7, 13]);
  const merged: Int16Array = new Int16Array(3);
  for (let index: number = 0; index < 3; index++) {
    merged[index] = leftSamples[index] + rightSamples[index];
  }
  return merged;
}
