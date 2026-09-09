# @security/no-cycle

Static repair references: 11. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_007b408a3792e734`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// ComponentA.ets
import {} from './ComponentB.ets';

// ComponentB.ets
import {} from './ComponentA.ets';
```

### Repair pattern

```arkts
// ComponentA.ets
import {} from './ComponentC.ets';

// ComponentB.ets
import {} from './ComponentA.ets';
```

### Rationale

循环依赖会影响代码的性能并增加调试的复杂性。

## Example 2: `pair_10e3357271cac952`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// Alpha.ets
import {} from './Beta.ets';

// Beta.ets
import {} from './Alpha.ets';
```

### Repair pattern

```arkts
// Alpha.ets
import {} from './Gamma.ets';

// Beta.ets
import {} from './Alpha.ets';
```

### Rationale

循环依赖破坏了模块的完整性和模块化设计原则。

## Example 3: `pair_2f46744c5e4fea29`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// Node1.ets
import {} from './Node2.ets';

// Node2.ets
import {} from './Node1.ets';
```

### Repair pattern

```arkts
// Node1.ets
import {} from './Node3.ets';

// Node2.ets
import {} from './Node1.ets';
```

### Rationale

循环依赖可能导致不一致的模块状态，破坏应用逻辑。

## Example 4: `pair_58af23d924871877`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// FileA.ets
import {} from './FileB.ets';

// FileB.ets
import {} from './FileA.ets';
```

### Repair pattern

```arkts
// FileA.ets
import {} from './FileC.ets';

// FileB.ets
import {} from './FileA.ets';
```

### Rationale

循环依赖会导致模块加载失败或导致内存泄漏。

## Example 5: `pair_7fc6b9aa8f67ce7c`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// X.ets
import {} from './Y.ets';

// Y.ets
import {} from './X.ets';
```

### Repair pattern

```arkts
// X.ets
import {} from './Z.ets';

// Y.ets
import {} from './X.ets';
```

### Rationale

循环依赖会导致代码的可维护性变差。

## Example 6: `pair_9351823a6eaba080`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// Element1.ets
import {} from './Element2.ets';

// Element2.ets
import {} from './Element1.ets';
```

### Repair pattern

```arkts
// Element1.ets
import {} from './Element4.ets';

// Element2.ets
import {} from './Element1.ets';
```

### Rationale

循环依赖破坏了代码的逻辑顺序，使得调试变得困难。

## Example 7: `pair_9eff4e6ac1cc5eba`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// FileX.ets
import {} from './FileY.ets';

// FileY.ets
import {} from './FileX.ets';
```

### Repair pattern

```arkts
// FileX.ets
import {} from './FileZ.ets';

// FileY.ets
import {} from './FileX.ets';
```

### Rationale

循环依赖会导致模块加载的速度变慢或失败。

## Example 8: `pair_a3634ace0fd93db9`

禁止 ETS 模块之间存在循环依赖。

### Triggering pattern

```arkts
import { LoopingHelper } from './no_cycle_helper_bad';

class CircularDashboard {
  private helper?: LoopingHelper;

  load(helper: LoopingHelper): void {
    this.helper = helper;
  }

  render(): string {
    const stats: Array<number> = [1, 3, 5];
    const sum: number = stats.reduce((acc: number, val: number) => acc + val, 0);
    return `${this.helper?.describe() ?? 'none'}-${sum}`;
  }
}

export class LoopingHelper {
  private readonly name: string;

  constructor(name: string) {
    this.name = name;
  }

  describe(): string {
    return `helper:${this.name}`;
  }
}

export function renderCircular(): string {
  const dashboard: CircularDashboard = new CircularDashboard();
  const helper: LoopingHelper = new LoopingHelper('cycle');
  dashboard.load(helper);
  return dashboard.render();
}
```

### Repair pattern

```arkts
class ReportAssembler {
  constructor(private readonly segments: Array<string>) {}

  assemble(): string {
    return this.segments.reduce((acc: string, next: string) => `${acc}-${next}`, 'root');
  }
}

export function buildRiskReport(): string {
  const assembler: ReportAssembler = new ReportAssembler(['metrics', 'policy', 'result']);
  return assembler.assemble();
}
```

### Rationale

默认场景 代码触发 @security/no-cycle：禁止 ETS 模块之间存在循环依赖。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 9: `pair_a4c8ad85a5c145df`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// Object1.ets
import {} from './Object2.ets';

// Object2.ets
import {} from './Object1.ets';
```

### Repair pattern

```arkts
// Object1.ets
import {} from './Object3.ets';

// Object2.ets
import {} from './Object1.ets';
```

### Rationale

循环依赖可能导致应用程序的意外行为。

## Example 10: `pair_c608b45c8ebbad0b`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// PackageA.ets
import {} from './PackageB.ets';

// PackageB.ets
import {} from './PackageA.ets';
```

### Repair pattern

```arkts
// PackageA.ets
import {} from './PackageC.ets';

// PackageB.ets
import {} from './PackageA.ets';
```

### Rationale

循环依赖使得代码的重用性变差，影响代码的模块化。

## Example 11: `pair_ce3efc059772f8d6`

该规则禁止使用循环依赖

### Triggering pattern

```arkts
// Module1.ets
import {} from './Module2.ets';

// Module2.ets
import {} from './Module1.ets';
```

### Repair pattern

```arkts
// Module1.ets
import {} from './Module3.ets';

// Module2.ets
import {} from './Module1.ets';
```

### Rationale

循环依赖可能导致模块加载的死锁状态。
