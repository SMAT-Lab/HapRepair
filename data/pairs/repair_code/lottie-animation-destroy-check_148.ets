import lottie from '@ohos/lottie';

@Entry
@Component
struct MissedDestroyFixed {
  private renderController: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.renderController)
      .width(180)
      .height(180)
      .onReady(() => {
        lottie.loadAnimation({
          container: this.renderController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'missed_destroy.json'
        });
      });
  }

  onPageHide() {
    lottie.destroy();
  }
}