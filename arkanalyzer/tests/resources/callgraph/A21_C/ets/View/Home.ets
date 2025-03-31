
import Brightness from '@system.brightness';


@Entry
@Component
export struct Home {
  select: number = 1;
  currentIndex: number = 0;

  build() {

      Column() {
        Text('主卧')
          .fontColor(Color.White)
          .fontSize(40)

        Stack({
          alignContent: Alignment.Center

        }) {
          Progress({ value: 40, total: 100, type: ProgressType.Ring })
            .size({ width: 250, height: 250 })
            .margin({ top: 100 })
            .style({ strokeWidth: 20 })

          Column() {
            Text("PM 2.5")
              .fontColor(Color.White)
              .fontSize(25)
              .margin(
                {
                  top: 70
                }
              )

            Text(this.pm25)
              .fontColor(Color.White)
              .fontSize(40)
              .margin(
                {
                  top: 30
                }
              )


            Text(this.airQuality)
              .fontColor(Color.White)
              .fontSize(25)
              .margin(
                {
                  top: 25
                }
              )


          }
        }
        Row() {
          Column() {
            Text(" 照明（LX）")
              .fontColor(Color.White)
              .fontSize(20)
              .width('33%')
              .margin({
                top: 150
              })
            Text("10")
              .fontColor(Color.White)
              .fontSize(20)
              .margin({
                right: 25
              })
          }

          Line()
            .width(2)
            .height(90)
            .backgroundColor('#F5F5F5')
            .margin(5)
            .margin(
              {
                top: 130
              }
            )

          Column() {
            Text('温度（℃）')
              .fontColor(Color.White)
              .fontSize(20)
              .margin({
                top: 150,
                left: 30
              })
            Text("10")
              .fontColor(Color.White)
              .fontSize(20)

          }

          Line()
            .width(2)
            .height(90)
            .backgroundColor('#F5F5F5')
            .margin(5)
            .margin(
              {
                top: 130
              }
            )

          Column() {
            Text(" 湿度（%）")
              .fontColor(Color.White)
              .fontSize(20)
              .width('33%')
              .margin({
                top: 150,
                left: 15
              })
            Text("10")
              .fontColor(Color.White)
              .fontSize(20)
              .margin({
                right: 10
              })
          }
        }
        Row() {
          Column() {
            Button() {
              Image($r("app.media.sleep"))
                .width(50)
                .height(50)
                .borderRadius(50)

            }
            .margin(
              {
                right: 75
              }
            )
            .backgroundColor('#03A89E')
            Text('睡眠')
              .fontColor(Color.White)
              .fontSize(20)
              .margin({
                right: 75
              })
          }
          .margin({
            top: 50
          })

          Column() {
            Button() {
              Image($r("app.media.LED"))
                .width(50)
                .height(50)
                .borderRadius(50)
            }
            .margin({
              right:30
            })
            .backgroundColor('#03A89E')
            Text('LED灯')
              .fontColor(Color.White)
              .fontSize(20)
              .margin({
                right:20
              })
          }
          .margin({
            top :50
          })

          Column() {
            Button() {
              Image($r("app.media.fengshan"))
                .width(50)
                .height(50)
                .borderRadius(50)
            }
            .margin({
              left:40
            })
            .backgroundColor('#03A89E')
            Text('风扇')
              .fontColor(Color.White)
              .fontSize(20)
              .margin({
                left:40
              })
          }
          .margin({
            top : 50
          })


        }

      }
    .height('100%')
    .backgroundColor('#03A89E')
    .padding({
      bottom: 10
            })


  }
}