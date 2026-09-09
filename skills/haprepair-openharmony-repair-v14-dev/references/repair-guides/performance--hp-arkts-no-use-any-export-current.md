# @performance/hp-arkts-no-use-any-export-current

Static repair references: 3. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_19994a07214e2c98`

避免使用export * 导出当前module中定义的类型和数据。

冷启动完成时延场景下，建议优先修改

### Triggering pattern

```arkts
class User {
  id?: number;
  name?: string;
}
// 当前文件 User.ets
export * from './User';
// 当前文件 User.ets
export * as XX from './User';
```

### Repair pattern

```arkts
export class User {
  id?: number;
  name?: string;
}
```

### Rationale

避免使用export * 导出当前module中定义的User类

## Example 2: `pair_74b34fb54c311348`

避免使用export * 导出当前module中定义的类型和数据。

冷启动完成时延场景下，建议优先修改

### Triggering pattern

```arkts
// 文件：Utility.ets
function calculateSum(a: number, b: number): number {
  return a + b;
}

function calculateDifference(a: number, b: number): number {
  return a - b;
}

// 错误地使用了 export * 导出当前模块的定义
export * from './Utility';
```

### Repair pattern

```arkts
// 文件：Utility.ets
export function calculateSum(a: number, b: number): number {
  return a + b;
}

export function calculateDifference(a: number, b: number): number {
  return a - b;
}
```

### Rationale

避免使用 export * 导出当前模块中定义的函数。应直接导出具体的函数名称，以增强代码的明确性和可维护性。

## Example 3: `pair_ee383ed4eee55ab2`

避免使用export * 导出当前module中定义的类型和数据。

冷启动完成时延场景下，建议优先修改

### Triggering pattern

```arkts
// 文件：Constants.ets
const MAX_LIMIT = 100;
const MIN_LIMIT = 1;

// 错误地使用了 export * 导出当前模块的定义
export * from './Constants';
```

### Repair pattern

```arkts
// 文件：Constants.ets
export const MAX_LIMIT = 100;
export const MIN_LIMIT = 1;
```

### Rationale

避免使用 export * 导出当前模块中定义的常量。直接导出具体的常量名称，可以使代码更清晰，便于他人理解和使用。
