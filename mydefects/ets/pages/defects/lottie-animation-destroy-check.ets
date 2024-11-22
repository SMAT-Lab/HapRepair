import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

const ANIMATION_PATH: string = 'media/funnyAnimation.json';

@Entry
@Component
struct LottieAnimation2 {
  private animationController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem?: AnimationItem;

  build() {
    Column() {
      Canvas(this.animationController)
        .width(200)
        .height(200)
        .backgroundColor(Color.White)
        .onReady(() => {
          this.animationItem = lottie.loadAnimation({
            container: this.animationController,
            renderer: 'canvas',
            loop: true,
            autoplay: true,
            path: ANIMATION_PATH,
          });
        })

      Button('Stop Animation')
        .onClick(() => {
          if (this.animationItem) {
            this.animationItem.stop();
          }
        })
        .padding(10)
        .backgroundColor(Color.Blue)
        .fontColor(Color.White)
    }
  }
}
