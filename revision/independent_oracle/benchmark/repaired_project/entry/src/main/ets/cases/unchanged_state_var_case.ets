@Entry
@Component
struct StaticActionLabel {
  private actionLabel: string = 'Continue';

  build() {
    Button(this.actionLabel)
  }
}
