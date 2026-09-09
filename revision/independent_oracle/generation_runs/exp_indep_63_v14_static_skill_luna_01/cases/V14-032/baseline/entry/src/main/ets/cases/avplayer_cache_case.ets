import media from '@ohos.multimedia.media';

@Entry
@Component
struct VoiceMemoPreview {
  private previewPlayer?: media.AVPlayer;
  @State status: string = 'Preparing preview';

  aboutToAppear(): void {
    media.createAVPlayer().then((player: media.AVPlayer) => {
      this.previewPlayer = player;
      this.status = 'Preview ready';
    });
  }

  aboutToDisappear(): void {
    if (this.previewPlayer) {
      this.previewPlayer.release();
    }
    this.previewPlayer = undefined;
  }

  build() {
    Column({ space: 10 }) {
      Text('Voice memo')
      Text(this.status)
    }
  }
}
