@Entry
@Component
struct StaticActionLabel {
  @State actionLabel: string = 'Continue';

  build() {
    Button(this.actionLabel)
  }
}
