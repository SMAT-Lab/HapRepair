import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

// 动画播放的起始帧
const FRAME_START: number = 60; 
// 动画播放的终止帧
const FRAME_END: number = 120; 

@Entry
@Component
struct LottieAnimation2 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  // 动画名称
  private politeChicky: string = 'politeChicky'; 
  // hap包内动画资源文件路径，仅支持json格式
  private politeChickyPath: string = 'media/politeChicky.json'; 
  private animateItem: AnimationItem | null = null;

  build() {
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .borderRadius(3)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name1',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
      .onClick(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name2',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
  }

  onPageHide(): void {
    lottie.destroy();
  }
}

