import lottie from '@ohos/lottie';

import { AnimationItem } from '@ohos/lottie';
@Entry
@Component
struct MultipleAnimationsComponentFixed {
  private animationContext: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private runningAnimations: AnimationItem[] = [];

  build() {
    Canvas(this.animationContext)
      .width(200)
      .height(200)
      .onReady(() => {
        const loadedAnim = lottie.loadAnimation({
          container: this.animationContext,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'multi_anim.json'
        });
        this.runningAnimations.push(loadedAnim);
      })
      .onDisAppear(() => {
        this.runningAnimations.forEach(anim => anim.destroy());
        this.runningAnimations = [];
      })
  }
}