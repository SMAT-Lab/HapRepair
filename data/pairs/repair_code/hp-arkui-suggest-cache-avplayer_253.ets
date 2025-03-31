import media from '@ohos.multimedia.media';

@Entry
@Component
struct AudioPlayerComponent {
  private avPlayer: media.AVPlayer | undefined = undefined;
  private avPlayerManager: AVPlayerManager = AVPlayerManager.getInstance();

  aboutToAppear(): void {
    // 使用缓存的 AVPlayer 实例
    this.avPlayerManager.switchPlayer();
    this.avPlayer = this.avPlayerManager.getCurrentPlayer();
    this.avPlayer?.prepare('media/audio1.mp3');
  }

  aboutToDisappear(): void {
    // 重置 AVPlayer 实例
    this.avPlayerManager.resetCurrentPlayer();
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

// AVPlayerManager 类同上例，不再重复
