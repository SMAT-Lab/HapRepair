class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}

function calculateSum(): number {
  let totalSum: number = 0;

  for (let outer: number = 0; outer < 5; outer++) {
    for (let inner: number = 1; inner < 10; inner++) {
      totalSum += (Time.info[inner - Time.start] % 2 === 0) ? outer * inner : 0;
    }
  }

  return totalSum;
}
