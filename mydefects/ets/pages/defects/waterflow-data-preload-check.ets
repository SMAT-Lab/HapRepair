import { WaterFlowDataSource } from '../data/WaterFlowDataSource'

@Entry
@Component
struct WaterFlowExample {
  public minSize: number = 100;
  public maxSize: number = 200;
  scroller: Scroller = new Scroller();
  dataSource: WaterFlowDataSource = new WaterFlowDataSource();
  private itemSizes: number[] = [];

  // 动态生成FlowItem的尺寸
  getSize(): number {
    let size = Math.random() * this.maxSize;
    return size > this.minSize ? size : this.minSize;
  }

  // 生成初始项的尺寸数组
  generateItemSizes() {
    for (let i = 0; i < 100; i++) {
      this.itemSizes.push(this.getSize());
    }
  }

  aboutToAppear() {
    this.generateItemSizes();
  }

  build() {
    Column() {
      WaterFlow() {
        LazyForEach(this.dataSource, (item: number) => {
          FlowItem() {
            FlowItemComponent({ item: item })
          }
          .width('100%')
          .height(this.itemSizes[item % 100])
          .backgroundColor(Color.Blue)
        }, (item: string) => item)
      }
      .cachedCount(2)
      .onReachEnd(() => {
        // 没有提前预加载新数据，滑动到结尾时才加载
        console.info("Reached end of water flow, loading more data...");
        setTimeout(() => {
          for (let i = 0; i < 50; i++) {
            this.dataSource.AddLastItem();
          }
        }, 1000);
      })
      .columnsTemplate("1fr 1fr")
      .columnsGap(10)
      .rowsGap(5)
      .backgroundColor(Color.LightGray)
      .width('100%')
      .height('100%')
    }
  }
}

@Reusable
@Component
struct FlowItemComponent {
  @Prop item: number;

  build() {
    Column() {
      Text(`Item: ${this.item}`)
        .fontSize(14)
        .padding(5)
      Image(`resources/item_${this.item % 5}.jpg`)
        .width('100%')
        .height(100)
        .objectFit(ImageFit.Fill)
    }
  }
}