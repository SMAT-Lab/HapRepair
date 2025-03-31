class Range {
  static bounds: number[] = [8, 16, 24, 32];
}

function detectBound(index: number): boolean {
  for (let b = 0; b < Range.bounds.length; b++) {
    if (Range.bounds[index] === Range.bounds[b]) { // 不变的值重复访问
      return true;
    }
  }
  return false;
}