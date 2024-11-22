let count = 100000;
let data: number[] = new Array(count);
data = new Array();
data[5000] = 42;  // 稀疏数组，只有一个元素赋值
data[75000] = 84; // 稀疏数组