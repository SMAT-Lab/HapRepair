import lottie from '@ohos/lottie';

@Entry
@Component
struct LottieExample1 {
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
        // 每次点击都加载新动画但未销毁之前的动画
        this.animationItem = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'animation.json'
        });
      })
      .onDisAppear(() => {
        lottie.destroy('non_existing_anim'); // 錯誤的动画名称或者只销毁一个动画
      });
  }
}
