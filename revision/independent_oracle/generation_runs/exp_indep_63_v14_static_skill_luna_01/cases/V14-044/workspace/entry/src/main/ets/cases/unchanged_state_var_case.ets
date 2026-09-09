@Entry
@Component
struct StaticActionLabel {
  actionLabel: string = 'Continue';

  build() {
    Button(this.actionLabel)
  }
}
