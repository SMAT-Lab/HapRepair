// 文件：Utility.ets
function calculateSum(a: number, b: number): number {
  return a + b;
}

function calculateDifference(a: number, b: number): number {
  return a - b;
}

// 错误地使用了 export * 导出当前模块的定义
export * from './Utility';
