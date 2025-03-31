// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';
import { GoodItems } from './data/DataEntry';

@Reusable
@Component
struct GoodItemComponent {
  @State introduce: string = ''
  @State price: string = ''
  @State numb: string = ''

  aboutToReuse(params: Record<string, ESObject>) {
    this.introduce = params.introduce
    this.price = params.price
    this.numb = params.numb
  }

  build() {
    Column() {
      Text(this.introduce)
        .fontSize(14)
        .padding({ left: 5, right: 5 })
        .margin({ top: 5 })
      Row() {
        Text('￥')
          .fontSize(10)
          .fontColor(Color.Red)
          .baselineOffset(-4)
        Text(this.price)
          .fontSize(16)
          .fontColor(Color.Red)
        Text(this.numb)
          .fontSize(10)
          .fontColor(Color.Gray)
          .baselineOffset(-4)
          .margin({ left: 5 })

      }
      .width('100%')
      .justifyContent(FlexAlign.SpaceBetween)
      .padding({ left: 5, right: 5 })
      .margin({ top: 15 })
    }
  }
}

@Entry
@Component
struct MyComponent{
  private data: MyDataSource = new MyDataSource([]);

  build() {
    Column() {
      LazyForEach(this.data, (item: GoodItems, index) => {
        GridItem() {
          GoodItemComponent({
            introduce: item.data.introduce,
            price: item.data.price,
            numb: item.data.numb,
          }).reuseId(item.numb)
        }
      }, (item: GoodItems) => item.index)
    }
  }
}