// 该规则检测使用lottie加载的动画是否都正确销毁。
//
// 当使用lottie加载动画时，一般需要先通过lottie.loadAnimation将动画加载到内存，动画执行完毕后需要在合适的时机（例如：onDisAppear，onPageHide，aboutToDisappear）通过调用animationItem的destroy方法将单个动画销毁或者调用lottie.destroy()方法将当前页面所有动画销毁，如果动画未被销毁就会造成资源浪费，影响应用性能体验。
//
// 内存优化场景下，建议优先修改。

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

