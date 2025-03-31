import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct LifecycleAnimationFixed {
  private animationSurface: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private activeAnimation?: AnimationItem;

  build() {
    Canvas(this.animationSurface)
      .width(150)
      .height(150)
      .onReady(() => {
        this.activeAnimation = lottie.loadAnimation({
          container: this.animationSurface,
          renderer: 'html',
          autoplay: true,
          path: 'lifecycle.json'
        });
      });
  }

  onPageHide() {
    this.activeAnimation?.destroy();
    this.activeAnimation = undefined;
  }
}