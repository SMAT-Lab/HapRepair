@Entry
@Component
struct App{
  build(){
    Column(){
      LazyForEach(this.data, item => {
          GridItem() {
            ProductGridItemComponent({
              uri: this.productData[item % 14].uri,
              title: this.productData[item % 14].title,
              labels: this.productData[item % 14].labels,
              price: item + 1,
              ratio: this.ratio // 你可能需要传递这个比例计算
            })
          }
          .width('95%')
          .height(300)
          .border({ width: 1, color: '#70808080', radius: 10 })
          .margin({ top: 3, bottom: 3 })
          .backgroundColor(Color.White)
          .padding({ bottom: 60 })
        }, item => item.toString())
    }

    .width('100%')
    .backgroundColor('#10000000')
  }
}

@Reusable
@Component
struct ProductGridItemComponent {
  @State uri: string = "";
  @State title: string = "";
  @State labels: string = "";
  @State price: number = 0;
  @State ratio: number = 0;

  constructor(uri: string, title: string, labels: string, price: number, ratio: number) {
    super()
    this.uri = uri;
    this.title = title;
    this.labels = labels;
    this.price = price;
    this.ratio = ratio;
  }

  build() {
    Column() {
      Image(this.uri)
        .width('100%')
        .height('40%')
        .objectFit(ImageFit.Contain)
        .margin({ bottom: 40 });

      Text(this.title)
        .fontSize(16 * this.ratio)
        .fontWeight(600)
        .margin({ bottom: 10, left: 10 });

      Row() {
        Text(this.labels)
          .fontSize(10 * this.ratio)
          .border({ width: 1, color: '#FA808080' })
          .margin({ bottom: 2, left: 10 })
          .padding(2);
      }
      .margin({ bottom: 2 });

      Text(`￥${this.price}`)
        .fontSize(16 * this.ratio)
        .fontColor(Color.Red)
        .margin({ left: 10 });
    }
    .alignItems(HorizontalAlign.Start);
  }
}
