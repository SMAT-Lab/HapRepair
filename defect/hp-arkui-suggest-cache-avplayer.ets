import media from '@ohos.multimedia.media';

@Entry
@Component
struct MyComponent {
  private avPlayer: media.AVPlayer | undefined = undefined;

  aboutToAppear(): void {
    // 每次进入页面时都会创建一个新的 AVPlayer 实例，未缓存
    media.createAVPlayer().then((ret) => {
      this.avPlayer = ret;
    })
  }

  aboutToDisappear(): void {
    // 每次离开页面时销毁 AVPlayer 实例
    if (this.avPlayer) {
      this.avPlayer.release();  // 释放播放器资源
    }
    this.avPlayer = undefined;
  }

  playVideo(): void {
    // 每次播放前都要等待创建播放器实例
    if (this.avPlayer) {
      this.avPlayer.setMediaSource('path/to/video/file');
      this.avPlayer.play();
    }
  }

  build() {
    Column() {
      Button("播放视频")
        .onClick(() => {
          this.playVideo();  // 点击时播放视频
        })
        .width('100%')
        .height(50)
    }
  }
}