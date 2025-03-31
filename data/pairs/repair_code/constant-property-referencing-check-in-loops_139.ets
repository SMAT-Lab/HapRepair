class Range {
  static bounds: number[] = [8, 16, 24, 32];
}

function detectBound(index: number): boolean {
  const checkBound = Range.bounds[index]; // 提取到循环外
  for (let b = 0; b < Range.bounds.length; b++) {
    if (checkBound === Range.bounds[b]) {
      return true;
    }
  }
  return false;
}