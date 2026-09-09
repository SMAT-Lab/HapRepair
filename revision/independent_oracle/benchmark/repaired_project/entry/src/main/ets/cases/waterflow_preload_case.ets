class PhotoDataSource implements IDataSource {
  private items: number[] = [1, 2, 3, 4, 5, 6];
  private listeners: DataChangeListener[] = [];

  totalCount(): number { return this.items.length; }
  getData(index: number): number { return this.items[index]; }

  registerDataChangeListener(listener: DataChangeListener): void {
    this.listeners.push(listener);
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    const index: number = this.listeners.indexOf(listener);
    if (index >= 0) {
      this.listeners.splice(index, 1);
    }
  }

  addLastItem(): void {
    const index: number = this.items.length;
    this.items.push(index + 1);
    this.listeners.forEach((listener: DataChangeListener) => {
      listener.onDataAdd(index);
    });
  }
}

@Entry
@Component
struct PhotoWaterFlow {
  private dataSource: PhotoDataSource = new PhotoDataSource();

  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          Text(`Photo ${item}`)
        }
        .onAppear(() => {
          if (item + 2 === this.dataSource.totalCount()) {
            this.dataSource.addLastItem();
          }
        })
      }, (item: number) => item.toString())
    }
    .cachedCount(4)
  }
}
