@Builder
function StatusChipBuilder(label: string) {
  Text(label)
    .padding(6)
    .backgroundColor(Color.Gray)
}

@Entry
@Component
struct StatusBoard {
  private labels: string[] = ['Ready', 'Busy'];

  build() {
    Row() {
      ForEach(this.labels, (label: string) => {
        StatusChipBuilder(label)
      }, (label: string) => label)
    }
  }
}
