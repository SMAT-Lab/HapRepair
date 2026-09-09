@Builder
function StatusChipBuilder(label: string) {
  Text(label)
    .padding(6)
    .backgroundColor(Color.Gray)
}

@Component
struct StatusChip {
  @Prop label: string;

  build() {
    StatusChipBuilder(this.label)
  }
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
