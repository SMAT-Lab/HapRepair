@Observed
class AudioState {
  mute: boolean = false;
}

@Entry
@Component
struct DemoH {
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
  @Link audio: AudioState;

  build() {
    Text('Volume Control')
      .opacity(this.audio.mute ? 0.5 : 1.0)
  }
}

@Component
struct MuteIndicator {
  @Link audio: AudioState;

  build() {
    Text('Mute Status')
      .opacity(this.audio.mute ? 1.0 : 0.5)
  }
}