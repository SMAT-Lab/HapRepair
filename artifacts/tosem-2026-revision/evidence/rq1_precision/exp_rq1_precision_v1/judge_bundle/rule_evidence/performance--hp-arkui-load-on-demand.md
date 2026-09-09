# @performance/hp-arkui-load-on-demand

Static repair references: 8. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1d9af0e3caf27d94`

建议使用按需加载。

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State arr: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

  build() {
    List() {
      ForEach(this.arr, (item: number) => {
        ListItem() {
          Text(`item value: ${item}`)
        }
      }, (item: number) => item.toString())
    }
    .width('100%')
    .height('100%')
  }
}
```

### Repair pattern

```arkts
import { MyDataSource } from './MyDataSource';
@Entry
@Component
struct MyComponent {
  private arr: MyDataSource = new MyDataSource()

  build() {
    List() {
      LazyForEach(this.arr, (item: string) => {
        ListItem() {
          Text(`item value: ${item}`)
        }
      }, (item: number) => item.toString())
    }
    .width('100%')
    .height('100%')
    .cachedCount(2)
  }
}
```

### Rationale

该代码中使用了 ForEach，不支持按需加载，可能导致性能问题。

## Example 2: `pair_4ba4d2a65f820ef2`

建议使用按需加载。

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  private items: number[] = Array.from({ length: 1000 }, (_, index) => index)

  build() {
    List() {
      ForEach(this.items, (item: number) => {
        ListItem() {
          Text(`Item: ${item}`)
        }
      })
    }
  }
}
```

### Repair pattern

```arkts
export class MyDataSource implements IDataSource {
  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  private items: number[] = [] //  = Array.from({ length: 1000 }, (_, index) => index)

  constructor() {
    for(let i=0; i< 1000; i++) {
      this.items.push(i)
    }
  }
  public getData(index: number): number {
    return this.items[index]
  }

  public totalCount(): number {
    return this.items.length
  }
}

@Entry
@Component
struct MyComponent {
  private dataSource: MyDataSource = new MyDataSource()

  build() {
    List() {
      LazyForEach(this.dataSource, (item: number) => {
        ListItem() {
          Text(`Item: ${item}`)
        }
      })
    }
    .width("100%")
    .height("100%")
    .cachedCount(2)
  }
}
```

### Rationale

使用了 ForEach 处理大量数据，会导致性能下降。

## Example 3: `pair_65240fbde0e676f5`

建议使用按需加载。

### Triggering pattern

```arkts
@Entry
@Component
struct AlphabetIndexerSample {
  scroller: Scroller = new Scroller()
  private value: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  private listValue: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  index: number = 0
  selectedColor: string = '#00ffff'
  popupColor: string = '#00ffff'
  selectedBackgroundColor: string = '#ff0000'
  popupBackground: string = '#ff0000'
  usingPopup: boolean = true
  itemSize: number = 13
  location: boolean = true

  build() {
    Row({ direction: FlexDirection.Column }) {
      Row() {
        NavigationBar({ title: 'AlphabetIndexer' })
      }.padding({ left: '3%' })

      Flex({ direction: this.location ? FlexDirection.Row : FlexDirection.RowReverse }) {
        List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
          LazyForEach(this.listValue, (item) => {
            ListItem() {
                Text('     ' + item)
                  .height(30)
                  .fontSize(16)
                  .width('100%')
                  .textAlign(TextAlign.Start)
              }

          })
        }
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
export class DataSource implements IDataSource{
  private dataArray: string[] = []
  private listeners: DataChangeListener[] = []

  constructor() {
    for(let i=0; i< 26; i++) {
      this.dataArray.push(i + 'a')
    }
  }

  public getData(index: number): string {
    return this.dataArray[index]
  }

  public totalCount(): number {
    return this.dataArray.length
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    if (this.listeners.indexOf(listener) < 0) {
      this.listeners.push(listener)
    }
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    const pos = this.listeners.indexOf(listener)
    if (pos >= 0) {
      this.listeners.splice(pos, 1)
    }
  }
}
@Entry
@Component
struct AlphabetIndexerSample {
  scroller: Scroller = new Scroller()
  private value: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  // private listValue: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  private listValue: DataSource = new DataSource()
  index: number = 0
  selectedColor: string = '#00ffff'
  popupColor: string = '#00ffff'
  selectedBackgroundColor: string = '#ff0000'
  popupBackground: string = '#ff0000'
  usingPopup: boolean = true
  itemSize: number = 13
  location: boolean = true

  build() {
    Column({ direction: FlexDirection.Column }) {
      Row() {
        NavigationBar({ title: 'AlphabetIndexer' })
      }.padding({ left: '3%' })

      Flex({ direction: this.location ? FlexDirection.Row : FlexDirection.RowReverse }) {
        List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
          LazyForEach(this.listValue, (item: string) => {
            ListItem() {
                Text('     ' + item)
                  .height(30)
                  .fontSize(16)
                  .width('100%')
                  .textAlign(TextAlign.Start)

            }
          })
        }
        .width("100%")
        .height("100%")
        .cachedCount(2)
      }
    }
  }
}
```

### Rationale

建议使用按需加载，使用LazyForEach来替代ForEach， 但是需要将数组变成IDataSource来实现按需加载

## Example 4: `pair_77e99e0e13a59184`

建议使用按需加载。

### Triggering pattern

```arkts
import { MyDataSource } from './MyDataSource';

@Reusable
@Component
struct ItemComponent {
  @State introduce: string = ''

  aboutToReuse(params: Record<string, ESObject>) {
    this.introduce = params.introduce
  }

  build() {
    Text(this.introduce)
      .fontSize(14)
      .padding({ left: 5, right: 5 })
      .margin({ top: 5 })
  }
}

@Entry
@Component
struct MyComponent {
  private data: number[] = [1,2,3,4,5,6,7]

  build() {
    List() {
      ForEach(this.data, (item: string) => {
        ListItem() {
          ItemComponent({ introduce: item }).reuseId(item)
        }
      }, (item: string) => item)
    }
    .width('100%')
    .height('100%')
  }
}
```

### Repair pattern

```arkts
import { MyDataSource } from './MyDataSource';

@Reusable
@Component
struct ItemComponent {
  @State introduce: string = ''

  aboutToReuse(params: Record<string, ESObject>) {
    this.introduce = params.introduce
  }

  build() {
    Text(this.introduce)
      .fontSize(14)
      .padding({ left: 5, right: 5 })
      .margin({ top: 5 })
  }
}

@Entry
@Component
struct MyComponent {
  private data: MyDataSource = new MyDataSource()

  build() {
    List() {
      LazyForEach(this.data, (item: string) => {
        ListItem() {
          ItemComponent({ introduce: item }).reuseId(item)
        }
      }, (item: string) => item)
    }
    .width('100%')
    .height('100%')
    .cachedCount(2)
  }
}
```

### Rationale

使用 ForEach 而不是 LazyForEach，造成性能问题。

## Example 5: `pair_841eadf5465bb3df`

建议使用按需加载。

### Triggering pattern

```arkts
@Entry
@Component
struct AlphabetIndexerSample {
  scroller: Scroller = new Scroller()
  private value: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  index: number = 0

  build() {
    Row({ direction: FlexDirection.Column }) {
      Row() {
        Navigation()
      }
      Flex({ direction: FlexDirection.Row }) {
        List() {
          ForEach(this.value, (item: string) => {
            ListItem() {
              Text(item)
            }
          }, (item:string)=>item)
        }
        .cachedCount(2)
        .width("100%")
        .height('100%')
      }
    }
  }
}
```

### Repair pattern

```arkts
export class DataSource implements IDataSource {
  private dataArray: string[] = []

  constructor() {
    for(let i = 0; i < 26; i++) {
      this.dataArray.push(String.fromCharCode(65 + i)) // A - Z
    }
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  public getData(index: number): string {
    return this.dataArray[index]
  }

  public totalCount(): number {
    return this.dataArray.length
  }
}

@Entry
@Component
struct AlphabetIndexerSample {
  private value: DataSource = new DataSource()

  build() {
    Column({ direction: FlexDirection.Column }) {
      Row() {
        Navigation() {
          
        }
      }
      Flex({ direction: FlexDirection.Row }) {
        List() {
          LazyForEach(this.value, (item: string) => {
            ListItem() {
              Text(item)
            }
          })
        }
        .cachedCount(2)
        .width("100%")
        .height("100%")
      }
    }
  }
}
```

### Rationale

此代码中使用了 ForEach，无法实现按需加载，可能导致性能低下。

## Example 6: `pair_bc502d6768f5b76f`

建议使用按需加载。

### Triggering pattern

```arkts
import { MyDataSource } from './MyDataSource';

@Reusable
@Component
struct ItemComponent {
  @State introduce: string = ''

  aboutToReuse(params: Record<string, ESObject>) {
    this.introduce = params.introduce
  }

  build() {
    Text(this.introduce)
      .fontSize(14)
      .padding({ left: 5, right: 5 })
      .margin({ top: 5 })
  }
}

@Entry
@Component
struct MyComponent {
  private data: number[] = [1,2,3,4,5,6,7]

  build() {
    List() {
      ForEach(this.data, (item: string) => {
        ListItem() {
          // 使用reuseId对不同的自定义组件实例分别标注复用组，以达到最佳的复用效果
          ItemComponent({ introduce: item }).reuseId(item)
        }
      }, (item: string) => item)
    }
    .width('100%')
    .height('100%')
  }
}
```

### Repair pattern

```arkts
import { MyDataSource } from './MyDataSource';

@Reusable
@Component
struct ItemComponent {
  @State introduce: string = ''

  aboutToReuse(params: Record<string, ESObject>) {
    this.introduce = params.introduce
  }

  build() {
    Text(this.introduce)
      .fontSize(14)
      .padding({ left: 5, right: 5 })
      .margin({ top: 5 })
  }
}

@Entry
@Component
struct MyComponent {
  private data: MyDataSource = new MyDataSource()

  build() {
    List() {
      LazyForEach(this.data, (item: string) => {
        ListItem() {
          // 使用reuseId对不同的自定义组件实例分别标注复用组，以达到最佳的复用效果
          ItemComponent({ introduce: item }).reuseId(item)
        }
      }, (item: string) => item)
    }
    .width('100%')
    .height('100%')
    .cachedCount(2)
  }
}
```

### Rationale

建议使用按需加载，使用LazyForEach来替代ForEach， 但是需要将数组变成IDataSource来实现按需加载

## Example 7: `pair_dc38ddc456fd7525`

建议使用按需加载。

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  private items: string[] = ["Apple", "Banana", "Cherry", "Date"]

  build() {
    List() {
      ForEach(this.items, (item: string) => {
        ListItem() {
          Text(item)
        }
      })
    }
  }
}
```

### Repair pattern

```arkts
export class MyStringDataSource implements IDataSource {
  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }
  private items: string[] = ["Apple", "Banana", "Cherry", "Date"]

  public getData(index: number): string {
    return this.items[index]
  }

  public totalCount(): number {
    return this.items.length
  }
}

@Entry
@Component
struct MyComponent {
  private dataSource: MyStringDataSource = new MyStringDataSource()

  build() {
    List() {
      LazyForEach(this.dataSource, (item: string) => {
        ListItem() {
          Text(item)
        }
      })
    }
    .width("100%")
    .height("100%")
    .cachedCount(2)
  }
}
```

### Rationale

使用 ForEach 而无法实现按需加载，造成性能浪费。

## Example 8: `pair_ee4f89f58521a5ae`

建议使用按需加载。

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State arr: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

  build() {
    List() {
      // List中建议使用LazyForEach
      ForEach(this.arr, (item: number) => {
        ListItem() {
          Text(`item value: ${item}`)
        }
      }, (item: number) => item.toString())
    }
    .width('100%')
    .height('100%')
  }
}
```

### Repair pattern

```arkts
import { MyDataSource } from './MyDataSource';
@Entry
@Component
struct MyComponent {
  private arr: MyDataSource = new MyDataSource()

  build() {
    List() {
      // List中建议使用LazyForEach
      LazyForEach(this.arr, (item: string) => {
        ListItem() {
          Text(`item value: ${item}`)
        }
      }, (item: number) => item.toString())
    }
    .width('100%')
    .height('100%')
    .cachedCount(2)
  }
}
```

### Rationale

建议使用按需加载，使用LazyForEach来替代ForEach， 但是需要将数组变成IDataSource来实现按需加载
