import lottie from '@ohos/lottie';

@Entry
@Component
struct ExampleComponentFixed {
  private myController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation?: AnimationItem;

  build() {
    Canvas(this.myController)
      .width(150)
      .height(150)
      .onReady(() => {
        this.animation = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          autoplay: true,
          path: 'example.json'
        });
      });
  }

  onDisAppear() {
    this.animation?.destroy();
    this.animation = null;
  }
}