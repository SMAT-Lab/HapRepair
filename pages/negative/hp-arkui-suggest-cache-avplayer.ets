// 建议缓存AVPlayer实例減少起播时延。
//
// 音视频起播速度慢的场景下，建议优先修改。

import media from '@ohos.multimedia.media';

@Entry
@Component
struct MyComponent{
  private avPlayer: media.AVPlayer | undefined = undefined;

  aboutToAppear(): void {
    // 页面创建时初始化AVPlayer实例
    media.createAVPlayer().then((ret) => {
      this.avPlayer = ret;
    });
  }

  aboutToDisappear(): void {
    // 离开页面时销毁AVPlayer实例
    if (this.avPlayer) {
      this.avPlayer.release();
    }
    this.avPlayer = undefined;
  }

  build() {
    // 组件布局
  }
}