@Component
struct StatusChip {
  @Prop label: string;

  build() {
    Text(this.label)
      .padding(6)
      .backgroundColor(Color.Gray)
  }
}

@Builder
function buildStatusChip(label: string) {
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
        buildStatusChip(label)
      }, (label: string) => label)
    }
  }
}
