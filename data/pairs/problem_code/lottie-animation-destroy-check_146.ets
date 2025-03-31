import lottie from '@ohos/lottie';

@Entry
@Component
struct MultipleAnimations {
  private controller: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animations: AnimationItem[] = [];

  build() {
    Canvas(this.controller)
      .width(180)
      .height(180)
      .onReady(() => {
        // 加载多个动画但未销毁
        for (let i = 0; i < 3; i++) {
          const anim = lottie.loadAnimation({
            container: this.controller,
            renderer: 'canvas',
            loop: true,
            autoplay: true,
            path: `animation_${i}.json`
          });
          this.animations.push(anim);
        }
      });
  }
}