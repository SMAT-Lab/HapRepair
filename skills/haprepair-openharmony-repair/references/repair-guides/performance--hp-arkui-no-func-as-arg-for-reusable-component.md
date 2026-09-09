# @performance/hp-arkui-no-func-as-arg-for-reusable-component

Static repair references: 2. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_8a61701af62dc0f4`

避免使用函数作为复用的自定义组件创建时的入参。

### Triggering pattern

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

@Entry
@Component
struct ProductList {
  private products: string[] = ['商品A', '商品B', '商品C'];
  private prices: Map<string, number> = new Map();

  // 模拟一个耗时的价格计算函数
  calculatePrice(productName: string): number {
    let price = 0;
    for (let i = 0; i < 100000; i++) {
      price += i;
    }
    return price;
  }

  aboutToAppear() {
    this.products.forEach((product) => {
      const price = this.calculatePrice(product);
      this.prices.set(product, price);
    });
  }

  build() {
    Column() {
      ForEach(this.products, (product: string) => {
        ProductItem({ name: product, price: this.prices.get(product) || 0 })
      }, (product: string) => product)
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
  prices: Map<string, number> = new Map<string, number>()

  aboutToReuse(params: Record<string, Object>): void {
    this.name = params.name as string;
    const prices = params.prices as Map<string, number>
    this.price = prices.get(this.name) || 0
  }

  build() {
    Column() {
      Text('产品名称：' + this.name)
      Text('价格：' + this.price)
    }
  }
}

@Entry
@Component
struct ProductList {
  private products: string[] = ['商品A', '商品B', '商品C'];
  private prices: Map<string, number> = new Map();

  // 模拟一个耗时的价格计算函数
  calculatePrice(productName: string): number {
    let price = 0;
    for (let i = 0; i < 100000; i++) {
      price += i;
    }
    return price;
  }

  aboutToAppear() {
    this.products.forEach((product) => {
      const price = this.calculatePrice(product);
      this.prices.set(product, price);
    });
  }

  build() {
    Column() {
      ForEach(this.products, (product: string) => {
        ProductItem({ name: product, prices: this.prices})
      }, (product: string) => product)
    }
  }
}
```

### Rationale

避免在创建复用组件时直接调用耗时函数

## Example 2: `pair_ccc68c5041f27ae5`

避免使用函数作为复用的自定义组件创建时的入参。

### Triggering pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

// 此处为复用的自定义组件
@Reusable
@Component
struct ChildComponent {
  @State desc: string = '';
  @State sum: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.desc = params.desc as string;
    this.sum = params.sum as number;
  }

  build() {
    Column() {
      Text('子组件' + this.desc)
        .fontSize(30)
        .fontWeight(30)
      Text('结果' + this.sum)
        .fontSize(30)
        .fontWeight(30)
    }
  }
}

@Entry
@Component
struct MyComponent{
  private data: MyDataSource = new MyDataSource();

  aboutToAppear(): void {
    for (let index = 0; index < 20; index++) {
      this.data.pushData(index.toString())
    }
  }

  // 真实场景的函数中可能存在未知的耗时操作逻辑，此处用循环函数模拟耗时操作
  count(): number {
    let temp: number = 0;
    for (let index = 0; index < 10000; index++) {
      temp += index;
    }
    return temp;
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.data, (item: string) => {
          ListItem() {
            // 此处sum参数是函数获取的，实际开发场景无法预料该函数可能出现的耗时操作，每次进行组件复用都会重复触发此函数的调用
            ChildComponent({ desc: item, sum: this.count() })
          }
          .width('100%')
          .height(100)
        }, (item: string) => item)
      }
      .height('100%')
      .width('100%')
    }
  }
}
```

### Repair pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

@Reusable
@Component
struct ChildComponent {
  @State desc: string = '';
  sum: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.desc = params.desc as string;
    this.sum = params.sum as number;
  }

  build() {
    Column() {
      Text('子组件' + this.desc)
        .fontSize(30)
        .fontWeight(30)
      Text('结果' + this.sum)
        .fontSize(30)
        .fontWeight(30)
    }
  }
}

@Entry
@Component
struct MyComponent{
  private data: MyDataSource = new MyDataSource();
  @State sum: number = 0;

  aboutToAppear(): void {
    for (let index = 0; index < 20; index++) {
      this.data.pushData(index.toString())
    }
    // 执行该异步函数
    this.count();
  }

  // 模拟耗时操作逻辑
  async count() {
    let temp: number = 0;
    for (let index = 0; index < 10000; index++) {
      temp += index;
    }
    // 将结果放入状态变量中
    this.sum = temp;
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.data, (item: string) => {
          ListItem() {
            // 子组件的传参通过状态变量进行
            ChildComponent({ desc: item, sum: this.sum })
          }
          .width('100%')
          .height(100)
        }, (item: string) => item)
      }
      .width('100%')
      .height('100%')
      .cachedCount(2)
    }
  }
}
```

### Rationale

复用组件ChildComponent时sum参数是函数获取的，实际开发场景无法预料该函数可能出现的耗时操作，每次进行组件复用都会重复触发此函数的调用，因此需要避免这样的函数调用
