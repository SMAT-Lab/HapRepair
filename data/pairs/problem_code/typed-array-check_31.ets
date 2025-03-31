const matrix: number[][] = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];
const result: number[] = new Array(matrix.length);
for (let i = 0; i < matrix.length; i++) {
  result[i] = matrix[i][i];
}