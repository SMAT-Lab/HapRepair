class Record {
  static data: number[] = [11, 12, 13, 14, 15];
}

function processValues(pointer: number, limit: number): number {
  let result: number = 1;
  for (let j = 0; j < limit; j++) {
      result *= Record.data[pointer - 1]; // 重复引用
  }
  return result;
}
