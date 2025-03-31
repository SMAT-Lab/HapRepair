import image from '@ohos.multimedia.image';
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';
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
    Swiper(this.swiperController) {
      LazyForEach(this.data, (item: MyObject, index?: number) => {
        // 源码文件，请以工程实际为准
        PhotoItem({ myIndex: index, dataSource: this.data })
      }, (item: MyObject) => item.id) // item唯一id
    }
    .cachedCount(this.cacheCount)
    .indicator(true)
    .loop(false)
    // 在OnAnimationStart接口回调中进行预加载资源的操作
    .onAnimationStart((index: number, targetIndex: number) => {
      if (targetIndex !== index) {
        try {
          // 获取resourceManager资源管理器
          const resourceMgr = this.context.resourceManager;
          // 获取rawfile文件夹下icon.png的ArrayBuffer
          let str = "item" + (targetIndex + this.cacheCount + 2) + ".jpg";
          resourceMgr.getRawFileContent(str).then((value) => {
            // 创建imageSource
            const imageSource = image.createImageSource(value.buffer);
            imageSource.createPixelMap().then((value) => {
              this.data.addData(targetIndex + this.cacheCount + 1, {
                description: "" + (targetIndex + this.cacheCount + 1),
                image: value
              })
            })
          })
        } catch (err) {
          console.log("error code" + err);
        }
      }
    })
    .width('100%')
    .height('100%')
  }
}