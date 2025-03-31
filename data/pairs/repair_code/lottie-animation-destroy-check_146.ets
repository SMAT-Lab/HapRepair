import lottie from '@ohos/lottie';

@Entry
@Component
struct MultipleAnimationsFixed {
  private controller: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animations: AnimationItem[] = [];

  build() {
    Canvas(this.controller)
      .width(180)
      .height(180)
      .onReady(() => {
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

  onPageHide() {
    this.animations.forEach(anim => anim.destroy());
    this.animations = [];
  }
}