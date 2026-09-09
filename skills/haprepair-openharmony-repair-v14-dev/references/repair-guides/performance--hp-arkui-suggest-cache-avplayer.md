# @performance/hp-arkui-suggest-cache-avplayer

Static repair references: 3. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_6d35b822977e3d73`

建议缓存AVPlayer实例減少起播时延。

音视频起播速度慢的场景下，建议优先修改

### Triggering pattern

```arkts
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
```

### Repair pattern

```arkts
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
```

### Rationale

原始代码中，每次页面出现时都会创建新的 AVPlayer 实例，并在页面消失时销毁。这会导致视频播放时产生起播时延。

通过引入 AVPlayerManager 单例类，缓存了两个 AVPlayer 实例。组件使用 AVPlayerManager 来获取和管理 AVPlayer，在页面出现时切换到缓存的播放器，避免频繁创建和销毁实例，从而减少起播时延。

## Example 2: `pair_7f1b2a7f02bc04ac`

建议缓存AVPlayer实例減少起播时延。

音视频起播速度慢的场景下，建议优先修改

### Triggering pattern

```arkts
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
```

### Repair pattern

```arkts
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
```

### Rationale

同样地，原始代码在页面出现和消失时创建和销毁 AVPlayer 实例，导致音频起播时延。通过使用 AVPlayerManager 缓存 AVPlayer 实例，避免了频繁的创建和销毁，提高了音频播放的启动速度。

## Example 3: `pair_a74cd493ea55dc54`

建议缓存AVPlayer实例減少起播时延。

音视频起播速度慢的场景下，建议优先修改

### Triggering pattern

```arkts
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
```

### Repair pattern

```arkts
import media from '@ohos.multimedia.media';

@Entry
@Component
struct MyComponent{
  private avPlayer: media.AVPlayer | undefined = undefined;
  private avPlayerManager: AVPlayerManager = AVPlayerManager.getInstance();

  aboutToAppear(): void {
    this.avPlayerManager.switchPlayer();
    this.avPlayer = this.avPlayerManager.getCurrentPlayer();
  }

  aboutToDisappear(): void {
    this.avPlayerManager.resetCurrentPlayer();
    this.avPlayer = undefined;
  }

  build() {
    // 组件布局
  }
}

class AVPlayerManager {
  private static instance?: AVPlayerManager;

  private player1?: media.AVPlayer;
  private player2?: media.AVPlayer;
  private currentPlayer?: media.AVPlayer;

  public static getInstance(): AVPlayerManager {
    if (!AVPlayerManager.instance) {
      AVPlayerManager.instance = new AVPlayerManager();
    }
    return AVPlayerManager.instance;
  }

  async AVPlayerManager() {
    this.player1 = await media.createAVPlayer();
    this.player2 = await media.createAVPlayer();
  }

  /**
   * 切换页面时切换AVPlayer实例
   */
  switchPlayer(): void {
    if (this.currentPlayer === this.player1) {
      this.currentPlayer = this.player2;
    } else {
      this.currentPlayer = this.player1;
    }
  }

  getCurrentPlayer(): media.AVPlayer | undefined {
    return this.currentPlayer;
  }

  /**
   * 使用reset方法重置AVPlayer实例
   */
  resetCurrentPlayer(): void {
    this.currentPlayer?.pause(() => {
      this.currentPlayer?.reset();
    });
  }
}
```

### Rationale

页面创建时初始化AVPlayer实例，使用缓存机制来缓存该实例
