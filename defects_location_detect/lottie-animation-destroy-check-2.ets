import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

const ANIMATION_PATH: string = 'media/sampleAnimation.json';

@Entry
@Component
struct LottieAnimationPage {
  private animationController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem?: AnimationItem;

  build() {
    Column() {
      Canvas(this.animationController)
        .width(300)
        .height(300)
        .backgroundColor(Color.Gray)
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
        .backgroundColor(Color.Red)
        .fontColor(Color.White)
    }
  }
}
