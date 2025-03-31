import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct FinishAnimation {
  private animationCtx: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationCtx)
      .width(160)
      .height(160)
      .onReady(() => {
        // 无动画销毁
        lottie.loadAnimation({
          container: this.animationCtx,
          renderer: 'svg',
          loop: false,
          autoplay: true,
          path: 'finish_anim.json'
        });
      });
  }
}