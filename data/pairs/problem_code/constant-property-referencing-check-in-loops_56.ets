class Database {
  static configurations: number[] = [22, 33, 44, 55, 66];
}

function updateSum(offset: number): number {
  let sum: number = 0;
  let key: number = 2;
  for (let i = 0; i < 4; i++) {
    sum += Database.configurations[key - offset];  // 多次访问相同值
  }
  return sum;
}
