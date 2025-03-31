
          Column() {
            Text(item.introduce)
              .fontSize(14)
              .padding({ left: 5, right: 5 })
              .margin({ top: 5 })
            Row() {
              Text('￥')
                .fontSize(10)
                .fontColor(Color.Red)
                .baselineOffset(-4)
              Text(item.price)
                .fontSize(16)
                .fontColor(Color.Red)
              Text(item.numb)
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
          .height(290)
        }