class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}
function getNum(num: number): number {
  /* Year has (12 * 29 =) 348 days at least */
  let total: number = 348;
  for (let year_0 = 1; year_0 <= 2024; year_0++) {
    // warning
    for(let month_0 = 1; month_0 <= 12; month_0++) {
      for(let day_0 = 1; day_0 <= 31; day_0++) {
        total += ((Time.info[num - Time.start] & day_0) !== 0) ? 1 : 0;
      }
    }

  }
  return total;
}