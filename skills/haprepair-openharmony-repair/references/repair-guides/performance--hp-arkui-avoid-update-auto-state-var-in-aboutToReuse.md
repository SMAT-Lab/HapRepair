# @performance/hp-arkui-avoid-update-auto-state-var-in-aboutToReuse

Static repair references: 3. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_2b23139ab39ae23a`

避免在aboutToReuse中对自动更新值的状态变量进行更新。

### Triggering pattern

```arkts
@Reusable
@Component
struct MessageItem {
  @Link sender: string;
  @State content: string = '';

  aboutToReuse(params: Record<string, Object>): void {
    this.sender = params.sender as string;
    this.content = params.content as string;
  }

  build() {
    Column() {
      Text('发件人：' + this.sender)
      Text('内容：' + this.content)
    }
  }
}

interface GeneratedTypeLiteralInterface_1 {
  sender: string;
  content: string;
}

@Entry
@Component
struct MessageList {
  private messages: Array<GeneratedTypeLiteralInterface_1> = [];

  aboutToAppear(): void {
    // 获取消息列表
    this.messages = [
      { sender: '张三', content: '你好' },
      { sender: '李四', content: '下午好' },
    ];
  }

  build() {
    Column() {
      List() {
        ForEach(this.messages, (message: GeneratedTypeLiteralInterface_1) => {
          ListItem() {
            MessageItem({ sender: message.sender, content: message.content })
          }
        }, (message: GeneratedTypeLiteralInterface_1) => message.content)
      }
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}
```

### Repair pattern

```arkts
@Reusable
@Component
struct MessageItem {
  @State sender: string = '';
  @State content: string = '';

  aboutToReuse(params: Record<string, Object>): void {
    this.sender = params.sender as string;
    this.content = params.content as string;
  }

  build() {
    Column() {
      Text('发件人：' + this.sender)
      Text('内容：' + this.content)
    }
  }
}

interface GeneratedTypeLiteralInterface_1 {
  sender: string;
  content: string;
}

@Entry
@Component
struct MessageList {
  private messages: Array<GeneratedTypeLiteralInterface_1> = [];

  aboutToAppear(): void {
    // 获取消息列表
    this.messages = [
      { sender: '张三', content: '你好' },
      { sender: '李四', content: '下午好' },
    ];
  }

  build() {
    Column() {
      List() {
        ForEach(this.messages, (message: GeneratedTypeLiteralInterface_1) => {
          ListItem() {
            MessageItem({ sender: message.sender, content: message.content })
          }
        }, (message:GeneratedTypeLiteralInterface_1) => message.content)
      }
      .width("100%")
      .height("100%")
    }
  }
}
```

### Rationale

将 sender 从 @Link 更改为 @State，以便在 aboutToReuse 方法中安全地对其进行更新。

## Example 2: `pair_6cd772386cca02b5`

避免在aboutToReuse中对自动更新值的状态变量进行更新。

### Triggering pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

// 此处为复用的自定义组件
@Reusable
@Component
struct ItemComponent {
  @State desc: string = '';
  @State sum: number = 0;
  @Link avg: number;

  aboutToReuse(params: Record<string, Object>): void {
    this.desc = params.desc as string;
    this.sum = params.sum as number;
    this.avg = params.avg as number;
  }

  build() {
    Column() {
      Text('子组件' + this.desc)
        .fontSize(30)
        .fontWeight(30)
      Text('结果' + this.sum)
        .fontSize(30)
        .fontWeight(30)
      Text('平均值' + this.avg)
        .fontSize(30)
        .fontWeight(30)
    }
  }
}

@Entry
@Component
struct MyComponent {
  private data: MyDataSource = new MyDataSource();

  aboutToAppear(): void {
    for (let index = 0; index < 20; index++) {
      this.data.pushData(index.toString())
    }
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.data, (item: string) => {
          ListItem() {
            ItemComponent({ desc: item, sum: 0, avg: 0 })
          }
          .width('100%')
          .height(100)
        }, (item: string) => item)
      }
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
    .width('100%')
    .height('100%')
  }
}
```

### Repair pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

// 此处为复用的自定义组件
@Reusable
@Component
struct ItemComponent {
  @State desc: string = '';
  @State sum: number = 0;
  @State avg: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.desc = params.desc as string;
    this.sum = params.sum as number;
    this.avg = params.avg as number;
  }

  build() {
    Column() {
      Text('子组件' + this.desc)
        .fontSize(30)
        .fontWeight(30)
      Text('结果' + this.sum)
        .fontSize(30)
        .fontWeight(30)
      Text('平均值' + this.avg)
        .fontSize(30)
        .fontWeight(30)
    }
  }
}

@Entry
@Component
struct MyComponent {
  private data: MyDataSource = new MyDataSource();

  aboutToAppear(): void {
    for (let index = 0; index < 20; index++) {
      this.data.pushData(index.toString())
    }
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.data, (item: string) => {
          ListItem() {
            ItemComponent({ desc: item, sum: 0, avg: 0 })
          }
          .width('100%')
          .height(100)
        }, (item: string) => item)
      }
      .width('100%')
      .height('100%')
      .cachedCount(2)
    }
    .width('100%')
    .height('100%')
  }
}
```

### Rationale

避免在aboutToReuse中对自动更新值的状态变量进行更新，因此需要将avg改为@State avg: number = 0;

## Example 3: `pair_cbd017714f96ea4e`

避免在aboutToReuse中对自动更新值的状态变量进行更新。

### Triggering pattern

```arkts
@Reusable
@Component
struct ProductItem {
  @State name: string = '';
  @Link price: number;

  aboutToReuse(params: Record<string, Object>): void {
    this.name = params.name as string;
    this.price = params.price as number;
  }

  build() {
    Column() {
      Text('产品名称：' + this.name)
      Text('价格：' + this.price)
    }
  }
}

interface GeneratedTypeLiteralInterface_1 {
  name: string;
  price: number;
}

@Entry
@Component
struct ProductList {
  private products: Array<GeneratedTypeLiteralInterface_1> = [];

  aboutToAppear(): void {
    // 初始化产品列表
    this.products = [
      { name: '商品A', price: 100 },
      { name: '商品B', price: 200 },
    ];
  }

  build() {
    Column() {
      List() {
        ForEach(this.products, (item: GeneratedTypeLiteralInterface_1) => {
          ListItem() {
            ProductItem({ name: item.name, price: item.price })
          }
        }, (item:GeneratedTypeLiteralInterface_1)=>item.name)
      }
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}
```

### Repair pattern

```arkts
@Reusable
@Component
struct ProductItem {
  @State name: string = '';
  @State price: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.name = params.name as string;
    this.price = params.price as number;
  }

  build() {
    Column() {
      Text('产品名称：' + this.name)
      Text('价格：' + this.price)
    }
  }
}

interface GeneratedTypeLiteralInterface_1 {
  name: string;
  price: number;
}

@Entry
@Component
struct ProductList {
  private products: Array<GeneratedTypeLiteralInterface_1> = [];

  aboutToAppear(): void {
    // 初始化产品列表
    this.products = [
      { name: '商品A', price: 100 },
      { name: '商品B', price: 200 },
    ];
  }

  build() {
    Column() {
      List() {
        ForEach(this.products, (item: GeneratedTypeLiteralInterface_1) => {
          ListItem() {
            ProductItem({ name: item.name, price: item.price })
          }
        }, (item: GeneratedTypeLiteralInterface_1) => item.name)
      }
      .width("100%")
      .height("100%")
    }
  }
}
```

### Rationale

在 ProductItem 组件中，price 被声明为 @Link，但在 aboutToReuse 方法中对其进行了赋值操作，这是不推荐的。
