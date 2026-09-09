# @performance/hp-arkts-no-use-any-export-other

Static repair references: 3. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_21b54824d812b8cc`

避免使用export * 导出其他module中定义的类型和数据。

冷启动完成时延场景下，建议优先修改。

### Triggering pattern

```arkts
export * from './Product';
export * as XX from './Product';
class User {
  id?: number;
  name?: string;
}
```

### Repair pattern

```arkts
export { Product } from './Product';
class User {
  id?: number;
  name?: string;
}
```

### Rationale

避免使用export * 导出其他module中定义的类型和数据，改为按需导入{Product}

## Example 2: `pair_2960a4e61c5b1c64`

避免使用export * 导出其他module中定义的类型和数据。

冷启动完成时延场景下，建议优先修改。

### Triggering pattern

```arkts
// 文件：index.ets
export * from './Utilities';
export * as Utils from './Utilities';

class App {
  // 应用的主要逻辑
}
```

### Repair pattern

```arkts
// 文件：index.ets
export { calculateSum, calculateDifference } from './Utilities';

class App {
  // 应用的主要逻辑
}
```

### Rationale

避免使用 export * 导出其他模块中的类型和数据。应当明确导出需要使用的成员，例如使用 export { calculateSum, calculateDifference } 从 './Utilities' 导出具体的函数。这有助于减少不必要的代码加载，提高应用的启动性能。

## Example 3: `pair_b9a33b4c05aa7cc3`

避免使用export * 导出其他module中定义的类型和数据。

冷启动完成时延场景下，建议优先修改。

### Triggering pattern

```arkts
// 文件：index.ets
export * from './Constants';
export * as Consts from './Constants';

interface Config {
  // 配置接口定义
}
```

### Repair pattern

```arkts
// 文件：index.ets
export { MAX_VALUE, MIN_VALUE } from './Constants';

interface Config {
  // 配置接口定义
}
```

### Rationale

避免使用 export * 导出其他模块中的常量。应当按需导出具体的常量，如 MAX_VALUE 和 MIN_VALUE，以减少代码体积，优化性能。
