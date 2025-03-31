import media from '@ohos.multimedia.media';

@Entry
@Component
struct VideoPlayerComponent {
  private avPlayer: media.AVPlayer | undefined = undefined;
  private avPlayerManager: AVPlayerManager = AVPlayerManager.getInstance();

  aboutToAppear(): void {
    // 使用缓存的 AVPlayer 实例
    this.avPlayerManager.switchPlayer();
    this.avPlayer = this.avPlayerManager.getCurrentPlayer();
    this.avPlayer?.setDisplaySurface(this.$element('videoSurface'));
    this.avPlayer?.prepare('media/video1.mp4');
  }

  aboutToDisappear(): void {
    // 重置 AVPlayer 实例而非销毁
    this.avPlayerManager.resetCurrentPlayer();
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

class AVPlayerManager {
  private static instance?: AVPlayerManager;

  private player1?: media.AVPlayer;
  private player2?: media.AVPlayer;
  private currentPlayer?: media.AVPlayer;

  private constructor() {
    // 初始化 AVPlayer 实例
    media.createAVPlayer().then((player) => {
      this.player1 = player;
    });
    media.createAVPlayer().then((player) => {
      this.player2 = player;
    });
  }

  public static getInstance(): AVPlayerManager {
    if (!AVPlayerManager.instance) {
      AVPlayerManager.instance = new AVPlayerManager();
    }
    return AVPlayerManager.instance;
  }

  /**
   * 切换当前使用的 AVPlayer 实例
   */
  switchPlayer(): void {
    this.currentPlayer = this.currentPlayer === this.player1 ? this.player2 : this.player1;
  }

  getCurrentPlayer(): media.AVPlayer | undefined {
    return this.currentPlayer;
  }

  /**
   * 重置当前的 AVPlayer 实例
   */
  resetCurrentPlayer(): void {
    this.currentPlayer?.pause();
    this.currentPlayer?.reset();
  }
}
