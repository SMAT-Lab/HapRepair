import { displaySync } from '@kit.ArkGraphics2D';

@Entry
@Component
struct AnimationComponent {
  build() {
    Column() {
      Text("High Frame Rate Animation Example")
        .fontSize(20)
        .padding(10)

      Button("Set Fixed High Frame Rate")
        .onClick(() => {
          let sync = displaySync.create();
          // 在此场景中，我们不合理地将 expected、min、max 都设置为 120
          sync.setExpectedFrameRateRange({
            expected: 120,
            min: 120,
            max: 120,
          });
          console.info('Frame rate locked to 120 FPS for animation');
        })
        .padding(10)
        .backgroundColor(Color.Green)
        .fontColor(Color.White)

      Button("Set Dynamic Frame Rate")
        .onClick(() => {
          let sync = displaySync.create();
          // 一个合理的动态帧率范围
          sync.setExpectedFrameRateRange({
            expected: 60,
            min: 30,
            max: 120,
          });
          console.info('Dynamic frame rate set between 30 FPS and 120 FPS');
        })
        .padding(10)
        .backgroundColor(Color.Red)
        .fontColor(Color.White)
    }
    .padding(20)
  }
}
