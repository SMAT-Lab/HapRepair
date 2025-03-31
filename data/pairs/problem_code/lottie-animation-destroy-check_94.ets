import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct LifecycleAnimation {
  private animationSurface: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationSurface)
      .width(150)
      .height(150)
      .onReady(() => {
        // 没有销毁操作
        lottie.loadAnimation({
          container: this.animationSurface,
          renderer: 'html',
          autoplay: true,
          path: 'lifecycle.json'
        });
      });
  }
}