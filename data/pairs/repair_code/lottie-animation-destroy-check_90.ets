import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct AnimationComponentFixed {
  private renderContext: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animateItem?: AnimationItem;

  build() {
    Canvas(this.renderContext)
      .width(120)
      .height(120)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.renderContext,
          renderer: 'canvas',
          autoplay: true,
          path: 'anim.json'
        });
      });
  }

  onDeactivate() {
    this.animateItem?.destroy();
    this.animateItem = undefined;
  }
}