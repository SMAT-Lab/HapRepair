# @performance/hp-arkui-use-grid-layout-options

Static repair references: 3. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_a81fc0899aa0273f`

建议在指定位置时使用GridLayoutOptions提升Grid性能

### Triggering pattern

```arkts
@Entry
@Component
struct PhotoGallery {
  private photos: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 500; i++) {
      this.photos.push('Photo ' + i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.photos, (photo: string, index: number) => {
          if ((index + 1) % 7 === 0) {
            GridItem() {
              Image($r('app.media.' + photo))
                .width('100%')
                .height(200)
            }
            .columnStart(0)
            .columnEnd(4)
          } else {
            GridItem() {
              Image($r('app.media.' + photo))
                .width('100%')
                .height(100)
            }
          }
        }, (photo: string) => photo)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct PhotoGallery {
  private photos: string[] = [];
  private irregularData: number[] = [];
  private layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: this.irregularData,
    getItemSize: (index: number) => {
      if (this.irregularData.includes(index)) {
        return [3, 1];
      }
      return [1, 1];
    }
  };

  aboutToAppear() {
    for (let i = 1; i <= 500; i++) {
      this.photos.push('Photo ' + i);
      if ((i) % 7 === 0) {
        this.irregularData.push(i - 1);
      }
    }
  }

  build() {
    Column() {
      Grid(undefined, this.layoutOptions) {
        LazyForEach(this.photos, (photo: string) => {
          GridItem() {
            Image($r('app.media.' + photo))
              .width('100%')
              .height(100)
          }
        }, (photo: string) => photo)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height('100%')
      .width('100%')
      .cachedCount(2)
    }
  }
}
```

### Rationale

原代码中，GridItem 使用了 columnStart 和 columnEnd 来控制特殊项的跨度，存在性能问题。通过使用 GridLayoutOptions 的 irregularIndexes 和 getItemSize 方法，可以更高效地管理不规则的网格项，提升性能。

## Example 2: `pair_b3e0ed789dde95bd`

建议在指定位置时使用GridLayoutOptions提升Grid性能

### Triggering pattern

```arkts
@Entry
@Component
struct ProductGrid {
  private products: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 1000; i++) {
      this.products.push('Product ' + i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.products, (item: string, index: number) => {
          if (index % 5 === 0) {
            GridItem() {
              Text(item)
                .fontSize(16)
                .width('100%')
                .height(120)
                .backgroundColor(Color.LightGray)
            }
            .columnStart(0)
            .columnEnd(3)
          } else {
            GridItem() {
              Text(item)
                .fontSize(16)
                .width('100%')
                .height(80)
                .backgroundColor(Color.White)
            }
          }
        }, (item: string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ProductGrid {
  private products: string[] = [];
  private irregularData: number[] = [];
  private layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: this.irregularData,
    getItemSize: (index: number) => {
      if (this.irregularData.includes(index)) {
        return [2, 1];
      }
      return [1, 1];
    }
  };

  aboutToAppear() {
    for (let i = 1; i <= 1000; i++) {
      this.products.push('Product ' + i);
      if ((i - 1) % 5 === 0) {
        this.irregularData.push(i - 1);
      }
    }
  }

  build() {
    Column() {
      Grid(undefined, this.layoutOptions) {
        LazyForEach(this.products, (item: string) => {
          GridItem() {
            Text(item)
              .fontSize(16)
              .width('100%')
              .height(80)
              .backgroundColor(Color.White)
          }
        }, (item: string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .height('100%')
      .width('100%')
      .cachedCount(2)
    }
  }
}
```

### Rationale

在原始代码中，GridItem 使用了 columnStart 和 columnEnd 来指定位置和跨度，这在大量数据下可能导致性能问题。通过使用 GridLayoutOptions，可以预先定义网格的布局和不规则项，提高 Grid 的渲染性能。

## Example 3: `pair_c6700273ce116994`

建议在指定位置时使用GridLayoutOptions提升Grid性能

### Triggering pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

@Reusable
@Component
struct TextItem {
  @State item: string = "";

  build() {
    Text(this.item)
      .fontSize(16)
      .backgroundColor(0xF9CF93)
      .width('100%')
      .height(80)
      .textAlign(TextAlign.Center)
      .onClick(() => {
        this.item = 'click';
      })
  }
}

@Entry
@Component
struct MyComponent{
  private datasource: MyDataSource = new MyDataSource();
  scroller: Scroller = new Scroller();

  aboutToAppear() {
    for (let i = 1; i <= 2000; i++) {
      this.datasource.pushData(i + '');
    }
  }

  build() {
    Column({ space: 5 }) {
      Text('Use columnStart and columnEnd to set the GridItem size').fontColor(0xCCCCCC).fontSize(9).width('90%')
      Grid(this.scroller) {
        LazyForEach(this.datasource, (item: string, index: number) => {
          if ((index % 4) === 0) {
            GridItem() {
              TextItem({ item: item })
            }
            .columnStart(0).columnEnd(2)
          } else {
            GridItem() {
              TextItem({ item: item })
            }
          }
        }, (item: string) => item)
      }
      .cachedCount(1)
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .width('90%')
      .height('40%')

      Button("scrollToIndex:1900").onClick(() => {
        this.scroller.scrollToIndex(1900);
      })
    }.width('100%')
    .margin({ top: 5 })
  }
}
```

### Repair pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

@Reusable
@Component
struct TextItem {
  @State item: string = "";

  build() {
    Text(this.item)
      .fontSize(16)
      .backgroundColor(0xF9CF93)
      .width('100%')
      .height(80)
      .textAlign(TextAlign.Center)
      .onClick(() => {
        this.item = 'click';
      })
  }
}

@Entry
@Component
export struct MyComponent{
  private datasource: MyDataSource = new MyDataSource();
  scroller: Scroller = new Scroller();
  private irregularData: number[] = [];
  layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: this.irregularData,
  };

  aboutToAppear() {
    for (let i = 1; i <= 2000; i++) {
      this.datasource.pushData(i + '');
      if ((i - 1) % 4 === 0) {
        this.irregularData.push(i - 1);
      }
    }
  }

  build() {
    Column({ space: 5 }) {
      Text('Set GridItem size using GridLayoutOptions').fontColor(0xCCCCCC).fontSize(9).width('90%')
      Grid(this.scroller, this.layoutOptions) {
        LazyForEach(this.datasource, (item: string, index: number) => {
          GridItem() {
            TextItem({ item: item })
          }
        }, (item: string) => item)
      }
      .cachedCount(1)
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .width('90%')
      .height('40%')

      Button("scrollToIndex:1900").onClick(() => {
        this.scroller.scrollToIndex(1900);
      })
    }.width('100%')
    .margin({ top: 5 })
  }
}
```

### Rationale

对于Grid(),建议给出layoutOptions来提高性能
