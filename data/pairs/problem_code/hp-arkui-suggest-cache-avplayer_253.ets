import media from '@ohos.multimedia.media';

@Entry
@Component
struct AudioPlayerComponent {
  private avPlayer: media.AVPlayer | undefined = undefined;

  aboutToAppear(): void {
    // 初始化 AVPlayer 实例
    media.createAVPlayer().then((player) => {
      this.avPlayer = player;
      this.avPlayer?.prepare('media/audio1.mp3');
    });
  }

  aboutToDisappear(): void {
    // 销毁 AVPlayer 实例
    if (this.avPlayer) {
      this.avPlayer.release();
    }
    this.avPlayer = undefined;
  }

  build() {
    Column() {
      Button('播放音频')
        .onClick(() => {
          this.avPlayer?.play();
        })
      Button('暂停音频')
        .onClick(() => {
          this.avPlayer?.pause();
        })
    }
  }
}
