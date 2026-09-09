# @performance/hp-arkui-use-local-var-to-replace-state-var

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0e24d28339636852`

建议使用临时变量替换状态变量

### Triggering pattern

```arkts
@Entry
@Component
struct ListManager {
  @State items: string[] = [];

  addItems(newItems: string[]) {
    this.items.push(newItems[0]);
    this.items.push(newItems[1]);
    this.items.push(newItems[2]);
  }

  build() {
    Column() {
      ForEach(this.items, (item) => {
        Text(item)
      })
      Button('添加项目')
        .onClick(() => this.addItems(['Item A', 'Item B', 'Item C']))
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ListManager {
  @State items: string[] = [];

  addItems(newItems: string[]) {
    let tempItems = [...this.items];
    tempItems.push(newItems[0]);
    tempItems.push(newItems[1]);
    tempItems.push(newItems[2]);
    this.items = tempItems;
  }

  build() {
    Column() {
      ForEach(this.items, (item) => {
        Text(item)
      }, (item:string)=>item)
      Button('添加项目')
        .onClick(() => this.addItems(['Item A', 'Item B', 'Item C']))
    }
  }
}
```

### Rationale

在方法中使用 tempItems 作为临时数组，复制原有的 items，然后在临时数组上进行 push 操作

## Example 2: `pair_2ff83b23896fed74`

建议使用临时变量替换状态变量

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State message: string = '';
  appendMsg(newMsg: String) {
      this.message += newMsg;
      this.message += ";";
      this.message += "<br/>";
  }
  build() {
    Button(this.message)
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State message: string = '';
  appendMsg(newMsg: String) {
      let message = this.message;
      message += newMsg;
      message += ";";
      message += "<br/>";
      this.message = message;
  }
  build() {
    Text(this.message)
  }
}
```

### Rationale

这里要多次对this.message这个状态变量进行操作，建议先用临时变量来存值，最后进行操作完后再重新给状态变量赋值

## Example 3: `pair_9201e8c36e8e8b21`

建议使用临时变量替换状态变量

### Triggering pattern

```arkts
@Entry
@Component
struct ShoppingCart {
  @State totalPrice: number = 0;

  addItem(price: number) {
    this.totalPrice += price;
    this.totalPrice += this.calculateTax(price);
    this.totalPrice += this.calculateShipping(price);
  }

  calculateTax(price: number): number {
    return price * 0.1;
  }

  calculateShipping(price: number): number {
    return 5;
  }

  build() {
    Column() {
      Text('总价：' + this.totalPrice)
        .fontSize(18)
      Button('添加商品')
        .onClick(() => this.addItem(100))
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ShoppingCart {
  @State totalPrice: number = 0;

  addItem(price: number) {
    let total = this.totalPrice;
    total += price;
    total += this.calculateTax(price);
    total += this.calculateShipping(price);
    this.totalPrice = total;
  }

  calculateTax(price: number): number {
    return price * 0.1;
  }

  calculateShipping(price: number): number {
    return 5;
  }

  build() {
    Column() {
      Text('总价：' + this.totalPrice)
        .fontSize(18)
      Button('添加商品')
        .onClick(() => this.addItem(100))
    }
  }
}
```

### Rationale

在 addItem() 方法中，使用局部变量 total 来累加总价的各项费用

## Example 4: `pair_b30df1e70922661d`

建议使用临时变量替换状态变量

### Triggering pattern

```arkts
@Entry
@Component
struct CounterComponent {
  @State count: number = 0;

  increment() {
    this.count += 1;
    this.count += 1;
    this.count += 1;
  }

  build() {
    Column() {
      Text('计数值：' + this.count)
        .fontSize(20)
      Button('增加计数')
        .onClick(() => this.increment())
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct CounterComponent {
  @State count: number = 0;

  increment() {
    let newCount = this.count;
    newCount += 1;
    newCount += 1;
    newCount += 1;
    this.count = newCount;
  }

  build() {
    Column() {
      Text('计数值：' + this.count)
        .fontSize(20)
      Button('增加计数')
        .onClick(() => this.increment())
    }
  }
}
```

### Rationale

在 increment() 方法中，使用局部变量 newCount 来累加计数值，避免多次直接修改状态变量。

## Example 5: `pair_f8bbfd20223adb97`

建议使用临时变量替换状态变量

### Triggering pattern

```arkts
@Entry
@Component
struct StringManipulator {
  @State result: string = '';

  manipulateString(input: string) {
    this.result = input.trim();
    this.result = this.result.toUpperCase();
    this.result = this.result + '!';
  }

  build() {
    Column() {
      Text('结果：' + this.result)
        .fontSize(16)
      Button('处理字符串')
        .onClick(() => this.manipulateString(' hello world '))
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct StringManipulator {
  @State result: string = '';

  manipulateString(input: string) {
    let tempResult = input.trim();
    tempResult = tempResult.toUpperCase();
    tempResult = tempResult + '!';
    this.result = tempResult;
  }

  build() {
    Column() {
      Text('结果：' + this.result)
        .fontSize(16)
      Button('处理字符串')
        .onClick(() => this.manipulateString(' hello world '))
    }
  }
}
```

### Rationale

在方法中使用 tempResult 来进行字符串的各项处理操作
