import lottie from '@ohos/lottie';

@Entry
@Component
struct IncompleteDestroy {
  private drawingController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.drawingController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.drawingController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          path: 'incomplete.json'
        });
      });
  }
}
