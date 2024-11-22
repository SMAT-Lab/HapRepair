interface GeneratedObjectLiteralInterface_1 {
  timeout: number;
  retries: number;
}

let config: GeneratedObjectLiteralInterface_1 = {
  timeout: 5000,
  retries: 3,
};

function fetchData() {
  // 直接访问全局变量 config
  console.log(`Fetching data with timeout: ${config.timeout}ms and retries: ${config.retries}`);

  // 模拟一些操作
  return config.timeout * config.retries;
}

fetchData();
