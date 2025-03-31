import lottie from '@ohos/lottie';

@Entry
@Component
struct LottieExample1Fixed {
  private myController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.myController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'animation.json'
        });
      })
      .onClick(() => {
        if (this.animationItem) {
          this.animationItem.destroy();  // 首先销毁现有动画实例
        }
        this.animationItem = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'animation.json'
        });
      });
  }

  onDisAppear() {
    lottie.destroy(); // 在页面消失时销毁所有相关动画
  }
}
