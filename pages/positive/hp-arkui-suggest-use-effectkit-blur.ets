// 导入图片处理模块
import image from "@ohos.multimedia.image";
// 导入图像效果模块
import effectKit from '@ohos.effectKit';

@Entry
@Component
struct MyComponent{
  // 是否显示全屏模态页面。静态模糊用
  @State isShowStaticBlur: boolean = false;
  // PixelMap实例
  @State pixelMap: image.PixelMap | undefined = undefined;
  // ImageSource实例
  imgSource: image.ImageSource | undefined = undefined;

  // 静态模糊
  async staticBlur() {
    // 获得当前Ability的Context
    let context = getContext(this);
    // 获取resourceManager对象
    let resourceMgr = context.resourceManager;
    // 获rawfile目录下的图片
    const fileData = await resourceMgr.getRawFileContent('startIcon.png');
    // 创建ArrayBuffer实例
    let buffer: ArrayBuffer = fileData.buffer.slice(0);
    // 创建图片源实例
    this.imgSource = image.createImageSource(buffer);
    // 创建像素的属性
    let opts: image.InitializationOptions = {
      // 是否可编辑
      editable: true,
      // 像素格式。3表示RGBA_8888
      pixelFormat: 3,
      // 创建图片大小
      size: {
        height: 4,
        width: 6
      }
    };
    // 创建PixelMap
    await this.imgSource.createPixelMap(opts).then((pixelMap: image.PixelMap) => {
      // 模糊半径
      const blurRadius = 1;
      // 创建Filter实例
      let headFilter = effectKit.createEffect(pixelMap);
      if (headFilter != null) {
        // 设置静态模糊。将模糊效果添加到效果链表中
        headFilter.blur(blurRadius);
        // 获取已添加链表效果的源图像的image.PixelMap
        headFilter.getEffectPixelMap().then((pixelMap: image.PixelMap) => {
          this.pixelMap = pixelMap;
        });
      }
    })
  }

  // 图片设置静态模糊的模态页面
  @Builder
  staticBlurBuilder() {
    Stack() {
      Image(this.pixelMap)
        .width('100%')
        .height('100%')
        .objectFit(ImageFit.Fill)
      Button('close')
        .fontSize(20)
        .onClick(() => {
          this.isShowStaticBlur = false;
        })
    }
    .width('100%')
    .height('100%')
  }

  build() {
    Column({ space: 10 }) {
      Button('静态模糊')
        .onClick(() => {
          this.isShowStaticBlur = true;
          // 设置静态模糊
          this.staticBlur();
        })
        .bindContentCover(this.isShowStaticBlur, this.staticBlurBuilder(), {
          // 全屏模态转场类型。上下切换动画
          modalTransition: ModalTransition.DEFAULT
        })
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
  }
}