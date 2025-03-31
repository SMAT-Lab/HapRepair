import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct MultipleAnimationsComponent {
  private animationContext: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationContext)
      .width(200)
      .height(200)
      .onReady(() => {
        // 未销毁多余动画
        lottie.loadAnimation({
          container: this.animationContext,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'multi_anim.json'
        });
      });
  }
}