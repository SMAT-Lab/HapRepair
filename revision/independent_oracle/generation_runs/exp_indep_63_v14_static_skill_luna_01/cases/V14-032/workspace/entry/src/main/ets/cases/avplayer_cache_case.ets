import media from '@ohos.multimedia.media';

class PreviewPlayerCache {
  private player?: media.AVPlayer;
  private pendingPlayer?: Promise<media.AVPlayer>;

  acquire(): Promise<media.AVPlayer> {
    if (this.player) {
      return Promise.resolve(this.player);
    }
    if (!this.pendingPlayer) {
      this.pendingPlayer = media.createAVPlayer().then((player: media.AVPlayer) => {
        this.player = player;
        this.pendingPlayer = undefined;
        return player;
      }).catch((error) => {
        this.pendingPlayer = undefined;
        throw error;
      });
    }
    return this.pendingPlayer;
  }

  reset(): void {
    this.player?.reset();
  }
}

const previewPlayerCache: PreviewPlayerCache = new PreviewPlayerCache();

@Entry
@Component
struct VoiceMemoPreview {
  private previewPlayer?: media.AVPlayer;
  private appearanceId: number = 0;
  @State status: string = 'Preparing preview';

  aboutToAppear(): void {
    const currentAppearanceId: number = ++this.appearanceId;
    this.status = 'Preparing preview';
    previewPlayerCache.acquire().then((player: media.AVPlayer) => {
      if (currentAppearanceId !== this.appearanceId) {
        return;
      }
      this.previewPlayer = player;
      this.status = 'Preview ready';
    }).catch(() => {
      if (currentAppearanceId === this.appearanceId) {
        this.status = 'Preview unavailable';
      }
    });
  }

  aboutToDisappear(): void {
    ++this.appearanceId;
    previewPlayerCache.reset();
    this.previewPlayer = undefined;
  }

  build() {
    Column({ space: 10 }) {
      Text('Voice memo')
      Text(this.status)
    }
  }
}
