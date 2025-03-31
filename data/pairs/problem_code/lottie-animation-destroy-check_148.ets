import lottie from '@ohos/lottie';

@Entry
@Component
struct MissedDestroy {
  private renderController: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.renderController)
      .width(180)
      .height(180)
      .onReady(() => {
        // 未销毁动画
        lottie.loadAnimation({
          container: this.renderController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'missed_destroy.json'
        });
      });
  }
}