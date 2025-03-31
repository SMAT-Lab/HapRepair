import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct PersistentAnimation {
  private animationCtx: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationCtx)
      .width(100)
      .height(100)
      .onReady(() => {
        // 页面隐藏时未执行销毁
        lottie.loadAnimation({
          container: this.animationCtx,
          renderer: 'canvas',
          autoplay: true,
          path: 'persistent.json'
        });
      });
  }
}