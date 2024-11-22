import { displaySync } from '@kit.ArkGraphics2D';

@Entry
@Component
struct MyComponent {
  build() {
    Column() {
      Text("Frame Rate Control Example")
        .fontSize(20)
        .padding(10)

      Button("Set High Frame Rate")
        .onClick(() => {
          let sync = displaySync.create();
          sync.setExpectedFrameRateRange({
            expected: 120,
            min: 120,
            max: 120,
          });
          console.info('Frame rate locked to 120 FPS');
        })
        .padding(10)
        .backgroundColor(Color.Blue)
        .fontColor(Color.White)
    }
    .padding(20)
  }
}