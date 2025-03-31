const matrix = [
  new Float64Array([1, 2, 3]),
  new Float64Array([4, 5, 6]),
  new Float64Array([7, 8, 9])
];
const result = new Float64Array(matrix.length);
for (let i = 0; i < matrix.length; i++) {
  result[i] = matrix[i][i];
}