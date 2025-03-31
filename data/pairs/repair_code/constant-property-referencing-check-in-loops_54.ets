class Record {
  static data: number[] = [11, 12, 13, 14, 15];
}

function processValues(pointer: number, limit: number): number {
  let result: number = 1;
  const fixedValue = Record.data[pointer - 1]; // 提取到循环外部
  for (let j = 0; j < limit; j++) {
    result *= fixedValue;
  }
  return result;
}
