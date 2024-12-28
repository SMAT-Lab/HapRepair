


import curves from '@ohos.curves';
import router from '@ohos.router';

@Entry
@Component
struct Logo {
  @State opacityValue: number = 0;
  @State scaleValue: number = 0;
  private rightPath: string = 'M319.5 128.1 c103.5 0 187.5 84 187.5 187.5 v15 a172.5 172.5 0 0 3 -172.5 172.5 H198 a36 36 0 0 3 -13.8 -1 207 207 0 0 0 87 -372 h48.3 z';
  private rightBottomPath: string = 'M270.6 128.1 h48.6 c51.6 0 98.4 21 132.3 54.6 a411 411 0 0 3 -45.6 123 c-25.2 45.6 -56.4 84 -87.6 110.4 a206.1 206.1 0 0 0 -47.7 -288 z';

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
      Shape() {
        Path()
          .commands('M162 128.7 a222 222 0 0 1 100.8 374.4 H198 a36 36 0 0 3 -36 -36')
          .fill(Color.White)

        Path()
          .commands(this.rightPath)
          .fill('none')
          .linearGradient({ angle: 30, colors: [['#c4ffa0', 0], ['#ffffff', 1]] })
          .clip(new Path().commands(this.rightPath))

        Path()
          .commands(this.rightBottomPath)
          .fill('none')
          .linearGradient({ angle: 50, colors: [['#8cc36a', 0.1], ['#b3eb90', 0.4], ['#ffffff', 0.7]] })
          .clip(new Path().commands(this.rightBottomPath))
      }
      .width(210)
      .height(210)
      .scale({ x: this.scaleValue, y: this.scaleValue })
      .opacity(this.opacityValue) // Set id to make transition interruptible
      .id('myShape')
      .transition(TransitionEffect.OPACITY.animation({ duration: 1000 }))
      .onAppear(() => {
        this.opacityValue = 1
        this.scaleValue = 1
      })

      Text($r("app.string.healthy_diet"))
        .margin({ top: 300 })
        .fontSize(26)
        .fontColor(Color.White)

      Text($r("app.string.logo_description"))
        .margin({ top: 4 })
        .fontSize(17)
        .fontColor(Color.White)
    }
    .width('100%')
    .height('100%')
    .linearGradient({ angle: 180, colors: [['#bde895', 0.1], ['#95de7f', 0.6], ['#7ab967', 1]] })
  }
}