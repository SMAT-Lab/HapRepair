import media from '@ohos.multimedia.media';

@Entry
@Component
struct VideoPlayerComponent {
  private avPlayer: media.AVPlayer | undefined = undefined;

  aboutToAppear(): void {
    // 页面创建时初始化 AVPlayer 实例
    media.createAVPlayer().then((player) => {
      this.avPlayer = player;
      this.avPlayer?.setDisplaySurface(this.$element('videoSurface'));
      this.avPlayer?.prepare('media/video1.mp4');
    });
  }

  aboutToDisappear(): void {
    // 离开页面时销毁 AVPlayer 实例
    if (this.avPlayer) {
      this.avPlayer.release();
    }
    this.avPlayer = undefined;
  }

  build() {
    Column() {
      // 视频显示区域
      VideoSurface()
        .id('videoSurface')
        .width('100%')
        .height(200)
      Button('播放视频')
        .onClick(() => {
          this.avPlayer?.play();
        })
    }
  }
}
