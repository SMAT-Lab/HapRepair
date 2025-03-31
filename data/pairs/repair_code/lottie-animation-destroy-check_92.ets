import lottie from '@ohos/lottie';

import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct PersistentAnimationFixed {
  private animationCtx: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationInstance?: AnimationItem;

  build() {
    Canvas(this.animationCtx)
      .width(100)
      .height(100)
      .onReady(() => {
        this.animationInstance = lottie.loadAnimation({
          container: this.animationCtx,
          renderer: 'canvas',
          autoplay: true,
          path: 'persistent.json'
        });
      });
  }

  onHidePage() {
    this.animationInstance?.destroy();
    this.animationInstance = undefined;
  }
}