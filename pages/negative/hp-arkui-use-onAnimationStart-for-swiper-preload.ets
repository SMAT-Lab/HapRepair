// 建议Swiper预加载机制搭配 OnAnimationStart 接口回调使用。
//
// 滑动丢帧场景下，建议优先修改。
import { MyDataSource } from './m/MyDataSource';
import { PhotoItem } from './component/ChildComponent';
import { MyObject } from './data/DataEntry';

@Entry
@Component
struct MyComponent{
  cacheCount: number = 1
  swiperController: SwiperController = new SwiperController();
  private data: MyDataSource = new MyDataSource([]);
  context = getContext(this);

  build() {
    // Swiper组件没有使用OnAnimationStart进行预加载
    Swiper(this.swiperController) {
      LazyForEach(this.data, (item: MyObject, index?: number) => {
        // 源码文件，请以工程实际为准
        PhotoItem({ myIndex: index, dataSource: this.data })
      }, (item: MyObject) => item.id)
    }
    .cachedCount(this.cacheCount)
    .indicator(true)
    .loop(false)
    .width('100%')
    .height('100%')
  }
}