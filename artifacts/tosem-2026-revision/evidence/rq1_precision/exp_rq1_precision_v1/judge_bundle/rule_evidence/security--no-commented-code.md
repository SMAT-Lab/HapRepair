# @security/no-commented-code

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_24caaaa8713695c2`

禁止以注释形式保留废弃或敏感代码。

### Triggering pattern

```arkts
@Entry
@Component
struct SyncPanel {
  @State status: string = 'idle';

  startSync(): void {
    this.status = 'running';
    // network.fetch('/sync/all');
  }

  build() {
    Column() {
      Text(`status: ${this.status}`)
      Button('Start')
        .onClick(() => {
          this.startSync();
          // this.status = 'failed';
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct SyncPanelClean {
  @State status: string = 'idle';

  startSync(): void {
    this.status = 'running';
  }

  build() {
    Column() {
      Text(`status: ${this.status}`)
      Button('Start')
        .onClick(() => {
          this.startSync();
        })
    }
  }
}
```

### Rationale

sync 代码触发 @security/no-commented-code：禁止以注释形式保留废弃或敏感代码。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_422040d5445c9f5d`

禁止以注释形式保留废弃或敏感代码。

### Triggering pattern

```arkts
@Component
struct PipelineMonitor {
  @State batches: Array<string> = [];

  record(batch: string): void {
    this.batches.push(batch);
    // hilog.warn(100, 'Pipeline', batch);
  }

  build() {
    Column() {
      Text(`count: ${this.batches.length}`)
      Button('Push Batch')
        .onClick(() => {
          this.record(`batch-${this.batches.length}`);
          // this.batches = [];
        })
    }
  }
}
```

### Repair pattern

```arkts
@Component
struct PipelineMonitorClean {
  @State batches: Array<string> = [];

  record(batch: string): void {
    this.batches.push(batch);
  }

  build() {
    Column() {
      Text(`count: ${this.batches.length}`)
      Button('Push Batch')
        .onClick(() => {
          this.record(`batch-${this.batches.length}`);
        })
    }
  }
}
```

### Rationale

pipeline 代码触发 @security/no-commented-code：禁止以注释形式保留废弃或敏感代码。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_6b7f43fcb6eb0089`

禁止以注释形式保留废弃或敏感代码。

### Triggering pattern

```arkts
@Entry
@Component
struct AuditBoard {
  @State logs: Array<string> = [];

  addRecord(record: string): void {
    this.logs.push(`${new Date().toUTCString()}: ${record}`);
    // hilog.info(1000, 'AuditBoard', record);
  }

  build() {
    Column() {
      Text(`records: ${this.logs.length}`)
        .fontSize(18)
      Button('Sync Ledger')
        .onClick(() => {
          this.addRecord('sync started');
          // this.logs = [];
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct AuditBoardClean {
  @State logs: Array<string> = [];

  addRecord(record: string): void {
    this.logs.push(`${new Date().toUTCString()}: ${record}`);
  }

  build() {
    Column() {
      Text(`records: ${this.logs.length}`)
        .fontSize(18)
      Button('Sync Ledger')
        .onClick(() => {
          this.addRecord('sync started');
        })
    }
  }
}
```

### Rationale

audit 代码触发 @security/no-commented-code：禁止以注释形式保留废弃或敏感代码。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_733ed01056c9e1e2`

禁止以注释形式保留废弃或敏感代码。

### Triggering pattern

```arkts
const nightlyJobs: Array<string> = ['fetch', 'filter', 'archive'];

class CommentedPipeline {
  private readonly stages: Array<string> = [];

  constructor(private readonly jobs: Array<string>) {}

  execute(): number {
    let processed: number = 0;
    this.jobs.map((job: string, idx: number) => {
      this.stages.push(`${idx}:${job}`);
      processed += idx + job.length;
    });
    return processed;
  }

  explain(): string {
    return this.stages.join(';');
  }
}

// console.info('temporary debug log left here intentionally');
// encryptSensitivePayload();

export function runCommentedPipeline(): string {
  const pipeline: CommentedPipeline = new CommentedPipeline(nightlyJobs);
  pipeline.execute();
  return pipeline.explain();
}
```

### Repair pattern

```arkts
const buildSteps: Array<string> = ['auth', 'sync', 'persist'];

class CommentFreePipeline {
  private readonly history: Array<string> = [];

  constructor(private readonly steps: Array<string>) {}

  runOnce(): string {
    let result: string = '';
    this.steps.forEach((step: string, index: number) => {
      const record: string = `${index}-${step}`;
      this.history.push(record);
      result += record + '|';
    });
    return result;
  }

  summarize(): string {
    return this.history.join(',');
  }
}

export function buildDeploymentPlan(): string {
  const pipeline: CommentFreePipeline = new CommentFreePipeline(buildSteps);
  const lastRun: string = pipeline.runOnce();
  return `${lastRun}=>${pipeline.summarize()}`;
}
```

### Rationale

默认场景 代码触发 @security/no-commented-code：禁止以注释形式保留废弃或敏感代码。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_8bd41e39114fbb1b`

禁止以注释形式保留废弃或敏感代码。

### Triggering pattern

```arkts
@Component
struct SensorWidget {
  @State temperature: number = 25;

  refreshMetrics(): void {
    this.temperature += 1;
    // this.temperature = sensor.read();
  }

  build() {
    Column() {
      Text(`Temp: ${this.temperature}`)
      Button('Calibrate')
        .onClick(() => {
          this.refreshMetrics();
          // this.temperature -= 5;
        })
    }
  }
}
```

### Repair pattern

```arkts
@Component
struct SensorWidgetClean {
  @State temperature: number = 25;

  refreshMetrics(): void {
    this.temperature += 1;
  }

  build() {
    Column() {
      Text(`Temp: ${this.temperature}`)
      Button('Calibrate')
        .onClick(() => {
          this.refreshMetrics();
        })
    }
  }
}
```

### Rationale

widget 代码触发 @security/no-commented-code：禁止以注释形式保留废弃或敏感代码。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
