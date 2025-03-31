import lottie from '@ohos/lottie';

@Entry
@Component
struct LottieExample2Fixed {
  private animationController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.animationController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.animationController,
          renderer: 'canvas',
          loop: false,
          autoplay: true,
          path: 'animation2.json'
        });
      });
  }

  onDisAppear() {
    if (this.animationItem) {
      this.animationItem.destroy(); // 清除动画确保内存释放
      this.animationItem = null;
    }
  }
}
