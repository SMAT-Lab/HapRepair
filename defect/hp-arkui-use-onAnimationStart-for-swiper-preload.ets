// 建议Swiper预加载机制搭配OnAnimationStart接口回调使用。
// 滑动丢帧场景下，建议优先修改。

import { MyImageDataSource } from '../data/MyImageDataSource';

@Entry
@Component
struct ImageSwiperComponent {
  cacheCount: number = 1;
  swiperController: SwiperController = new SwiperController();
  private imageData: MyImageDataSource = new MyImageDataSource([]);  // 模拟图片数据源

  build() {
    Column() {
      Text('Image Swiper').fontSize(24).padding(10)

      // Swiper组件没有使用OnAnimationStart进行预加载图片
      Swiper(this.swiperController) {
        LazyForEach(this.imageData, (imageUrl: string, index?: number) => {
          Image({ src: imageUrl })  // 显示每张图片
            .width('100%')
            .height(300)
        }, (imageUrl: string) => imageUrl)
      }
      .cachedCount(this.cacheCount)
      .indicator(true)
      .loop(true)
      .width('100%')
      .height(300)
    }
    .padding(20)
  }
}
