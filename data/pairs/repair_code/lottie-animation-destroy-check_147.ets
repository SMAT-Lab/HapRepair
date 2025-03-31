import lottie from '@ohos/lottie';

@Entry
@Component
struct SingleAnimationFixed {
  private context: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation: AnimationItem | undefined;

  build() {
    Canvas(this.context)
      .width(100)
      .height(100)
      .onReady(() => {
        this.animation = lottie.loadAnimation({
          container: this.context,
          renderer: 'svg',
          loop: false,
          autoplay: true,
          path: 'simple_anim.json'
        });
      });
  }

  aboutToDisappear() {
    this.animation?.destroy();
    this.animation = undefined;
  }
}