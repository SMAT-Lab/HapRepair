class Matrix {
  static dimensions: number[] = [5, 10, 15, 20];
}

function verifyDimension(target: number): boolean {
  for (let x = 0; x < Matrix.dimensions.length; x++) {
    if (Matrix.dimensions[target] === Matrix.dimensions[x]) { // 每次循环都访问相同值
      return true;
    }
  }
  return false;
}