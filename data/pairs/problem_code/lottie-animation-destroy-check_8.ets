import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

const FRAME_START: number = 60;
const FRAME_END: number = 120;

@Entry
@Component
struct LottieAnimation1 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private politeChicky: string = 'politeChicky';
  private politeChickyPath: string = 'media/politeChicky.json';
  private animateItem?: AnimationItem;

  build() {
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .backgroundColor(Color.Gray)
      .borderRadius(3)
      .onReady(() => {
        //告警
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: this.politeChicky,
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
  }
}

