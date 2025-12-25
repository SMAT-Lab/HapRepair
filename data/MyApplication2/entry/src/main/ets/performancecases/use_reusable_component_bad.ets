import { LazyForEach } from '@ohos/common';

// Simple demo to trigger @performance/hp-arkui-use-reusable-component:
// A complex GridItem inline in LazyForEach without using a reusable component.
// Below we refactor it using a @Reusable component, following the HomeCheck
// best-practice pattern.

@Reusable
@Component
struct GoodsItemCard {
  @State title: string = '';

  aboutToReuse(params: any) {
    // Parameters are provided as an object when used inside LazyForEach.
    if (params && params.title !== undefined) {
      this.title = params.title as string;
    }
  }

  build() {
    Column() {
      Text(this.title)
        .fontSize(16)
        .padding({ left: 5, right: 5 })
        .margin({ top: 5 })
      Row() {
        Text('￥')
          .fontSize(10)
          .fontColor(Color.Red)
          .baselineOffset(-4)
        Text('9.99')
          .fontSize(16)
          .fontColor(Color.Red)
        Text('库存充足')
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
    .borderRadius(10)
    .backgroundColor(Color.White)
    .clip(true)
    .width('100%')
    .height(120)
  }
}

@Entry
@Component
struct UseReusableComponentBad {
  private items: string[] = ['Apple', 'Banana', 'Orange', 'Grape'];

  build() {
    Column() {
      Grid() {
        LazyForEach(this.items, (item: string) => {
          GridItem() {
            GoodsItemCard({ title: item })
          }
        }, (item: string) => item)
      }
    }
  }
}

