class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}
function getNum(num: number): number {
  /* Year has (12 * 29 =) 348 days at least */
  let total: number = 348;
  const info = Time.info[num- Time.start];  
  for (let index: number = 0x8000; index > 0x8; index >>= 1) {
    if ((info & index) != 0) {
      total++;
    }
  }
  return total;
}