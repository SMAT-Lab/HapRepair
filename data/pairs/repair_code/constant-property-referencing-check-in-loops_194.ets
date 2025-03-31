class Matrix {
  static dimensions: number[] = [5, 10, 15, 20];
}

function verifyDimension(target: number): boolean {
  const targetDimension = Matrix.dimensions[target]; // 提取到循环外
  for (let x = 0; x < Matrix.dimensions.length; x++) {
    if (targetDimension === Matrix.dimensions[x]) {
      return true;
    }
  }
  return false;
}