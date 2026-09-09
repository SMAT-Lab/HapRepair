# @performance/hp-arkui-remove-redundant-state-var

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_017fda4d4c971917`

建议移除不关联UI组件的状态变量设置

### Triggering pattern

```arkts
@Entry
@Component
struct LogComponent {
  @State logMessages: string[] = [];

  addLog(message: string) {
    this.logMessages.push(message);
    // 执行日志记录操作
  }

  doSomething() {
    // 一些逻辑处理
  }

  build() {
    Column() {
      Button("执行操作")
        .onClick(() => this.doSomething())
      // `logMessages` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct LogComponent {
  logMessages: string[] = [];

  addLog(message: string) {
    this.logMessages.push(message);
    // 执行日志记录操作
  }

  doSomething() {
    // 一些逻辑处理
  }

  build() {
    Column() {
      Button("执行操作")
        .onClick(() => this.doSomething())
      // `logMessages` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }
}
```

### Rationale

状态变量 logMessages 未与任何 UI 组件关联，也未在 UI 调用的函数中使用。它仅在组件内部的方法中使用。因此，应移除 @State 装饰器，将其改为普通变量，以优化性能。

## Example 2: `pair_0749fa4f30d1c376`

建议移除不关联UI组件的状态变量设置

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State message: string = "";
  appendMsg(newMsg: String) : string {
    this.message += newMsg;
    return this.message;
  }
  build() {
    Column() {
      Stack() {
      }
      .backgroundColor("black")
      .width(200)
      .height(400)
      Button("move")
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MyComponent {

  appendMsg(newMsg: String) : string {
    this.message += newMsg;
    return this.message;
  }

  build() {
    Column() {
      Stack() {
      }
      .backgroundColor("black")
      .width(200)
      .height(400)
      Button("move")
    }
  }
}
```

### Rationale

message这个状态变量未使用，建议直接移除

## Example 3: `pair_9ebe6556719db781`

建议移除不关联UI组件的状态变量设置

### Triggering pattern

```arkts
@Entry
@Component
struct CounterComponent {
  @State counter: number = 0;

  incrementCounter() {
    this.counter += 1;
    // 内部逻辑处理，不关联 UI
  }

  resetCounter() {
    this.counter = 0;
    // 内部逻辑处理，不关联 UI
  }

  build() {
    Column() {
      Button("开始")
        .onClick(() => this.doSomething())
      Button("停止")
        .onClick(() => this.doSomething())
      // `counter` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }

  doSomething() {
    // 一些无关的逻辑
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct CounterComponent {
  counter: number = 0;

  incrementCounter() {
    this.counter += 1;
    // 内部逻辑处理，不关联 UI
  }

  resetCounter() {
    this.counter = 0;
    // 内部逻辑处理，不关联 UI
  }

  build() {
    Column() {
      Button("开始")
        .onClick(() => this.doSomething())
      Button("停止")
        .onClick(() => this.doSomething())
      // `counter` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }

  doSomething() {
    // 一些无关的逻辑
  }
}
```

### Rationale

状态变量 counter 未与任何 UI 组件关联，且未在 UI 调用的函数中使用。应移除 @State 装饰器，改为普通变量，减少不必要的状态管理开销。

## Example 4: `pair_a5e862ecb278bf90`

建议移除不关联UI组件的状态变量设置

### Triggering pattern

```arkts
@Entry
@Component
struct NotificationComponent {
  @State notificationCount: number = 0;

  incrementNotifications() {
    this.notificationCount += 1;
    // 逻辑处理，例如发送通知
  }

  doSomething() {
    // 执行其他逻辑
  }

  build() {
    Column() {
      Button("Send Notification")
        .onClick(() => this.doSomething())
      // notificationCount 未在 UI 中使用
    }
    .padding(10)
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct NotificationComponent {
  // notificationCount 未在 UI 中使用，移除状态变量
  notificationCount: number = 0;

  incrementNotifications() {
    this.notificationCount += 1;
    // 逻辑处理，例如发送通知
  }

  doSomething() {
    // 执行其他逻辑
  }

  build() {
    Column() {
      Button("Send Notification")
        .onClick(() => this.doSomething())
      // notificationCount 未在 UI 中使用
    }
    .padding(10)
  }
}
```

### Rationale

状态变量 notificationCount 虽然在组件内部被修改，但未与任何 UI 组件关联或展示。因此，移除 @State 标记，将其定义为普通变量，可以减少不必要的状态管理开销，优化性能。

## Example 5: `pair_e9f4a07e508cd16c`

建议移除不关联UI组件的状态变量设置

### Triggering pattern

```arkts
@Entry
@Component
struct DataFetcher {
  @State data: any = null;

  fetchData() {
    // 从 API 获取数据并赋值给 `data`
    this.data = { /* 获取的数据 */ };
  }

  build() {
    Column() {
      Button("加载数据")
        .onClick(() => this.doSomething())
      // `data` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }

  doSomething() {
    // 一些无关的逻辑
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct DataFetcher {
  data: any = null;

  fetchData() {
    // 从 API 获取数据并赋值给 `data`
    this.data = { /* 获取的数据 */ };
  }

  build() {
    Column() {
      Button("加载数据")
        .onClick(() => this.doSomething())
      // `data` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }

  doSomething() {
    // 一些无关的逻辑
  }
}
```

### Rationale

状态变量 data 未与任何 UI 组件关联，且未在 UI 调用的函数中使用。应移除 @State 装饰器，将其改为普通变量，以提高性能。
