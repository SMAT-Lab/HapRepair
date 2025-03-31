@Observed
class AudioState {
  mute: boolean = false;
}

@Entry
@Component
struct DemoHFixed {
  @State audio: AudioState = new AudioState();

  build() {
    Row() {
      VolumeControl({ audio: this.audio })
      MuteIndicator({ audio: this.audio })
      Text('Mute/Unmute')
        .onClick(() => {
          this.audio.mute = !this.audio.mute;
        })
    }
  }
}

@Component
struct VolumeControl {
  @Link @Watch('onMuteChange') audio: AudioState;
  @State localMute: boolean = false;

  onMuteChange() {
    this.localMute = this.audio.mute;
  }

  build() {
    Text('Volume Control')
      .opacity(this.localMute ? 0.5 : 1.0)
  }
}

@Component
struct MuteIndicator {
  @Link @Watch('onMuteChange') audio: AudioState;
  @State localMute: boolean = false;

  onMuteChange() {
    this.localMute = this.audio.mute;
  }

  build() {
    Text('Mute Status')
      .opacity(this.localMute ? 1.0 : 0.5)
  }
}