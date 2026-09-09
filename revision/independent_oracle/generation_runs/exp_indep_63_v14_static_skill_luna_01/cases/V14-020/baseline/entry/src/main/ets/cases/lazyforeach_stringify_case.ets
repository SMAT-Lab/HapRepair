interface QueueItem {
  id: string;
  label: string;
}

@Entry
@Component
struct QueueList {
  private items: QueueItem[] = [];

  build() {
    List() {
      LazyForEach(this.items, (item: QueueItem) => {
        ListItem() {
          Text(item.label)
        }
      }, (item: QueueItem) => JSON.stringify(item))
    }
  }
}
