class Database {
  static configurations: number[] = [22, 33, 44, 55, 66];
}

function updateSum(offset: number): number {
  let sum: number = 0;
  let key: number = 2;
  const configurationValue = Database.configurations[key - offset]; // 提取到循环外
  for (let i = 0; i < 4; i++) {
    sum += configurationValue;
  }
  return sum;
}
