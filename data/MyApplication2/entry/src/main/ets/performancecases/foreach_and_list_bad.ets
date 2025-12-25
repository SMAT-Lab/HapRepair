@Entry
@Component
struct ForeachAndListBad {
  private items: string[] = ['one', 'two', 'three'];

  build() {
    Column() {
      // Case 1: List without width/height + ForEach missing keyGenerator.
      List() {
        ForEach(this.items, (item: string, index: number) => {
          ListItem() {
            Text(item)
          }
        }, (item) => item)
      }

      // Case 2: List with initialIndex + ForEach missing keyGenerator.
      List({ initialIndex: 0 }) {
        ForEach(this.items, (item: string) => {
          ListItem() {
            Text(item)
          }
        }, (item) => item)
      }
    }
  }
}

