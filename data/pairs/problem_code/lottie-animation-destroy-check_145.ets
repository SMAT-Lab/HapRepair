import lottie from '@ohos/lottie';

@Entry
@Component
struct ExampleComponent {
  private myController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation?: AnimationItem;

  build() {
    Canvas(this.myController)
      .width(150)
      .height(150)
      .onReady(() => {
        // 未在页面消失时销毁动画
        this.animation = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          autoplay: true,
          path: 'example.json'
        });
      });
  }
}