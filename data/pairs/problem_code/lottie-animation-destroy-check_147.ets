import lottie from '@ohos/lottie';

@Entry
@Component
struct SingleAnimation {
  private context: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation: AnimationItem | undefined;

  build() {
    Canvas(this.context)
      .width(100)
      .height(100)
      .onReady(() => {
        // 无销毁逻辑
        this.animation = lottie.loadAnimation({
          container: this.context,
          renderer: 'svg',
          loop: false,
          autoplay: true,
          path: 'simple_anim.json'
        });
      });
  }
}