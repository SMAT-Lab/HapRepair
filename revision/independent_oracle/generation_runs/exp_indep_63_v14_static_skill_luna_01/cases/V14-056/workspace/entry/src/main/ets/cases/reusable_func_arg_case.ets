@Reusable
@Component
struct ComputedValueRow {
  @State value: number = 0;

  build() {
    Text(`${this.value}`)
  }
}

@Entry
@Component
struct ComputedValueList {
  private values: number[] = [2, 4, 6];

  computeTotal(): number {
    return this.values.reduce((sum: number, value: number) => sum + value, 0);
  }

  build() {
    const total: number = this.computeTotal();
    List() {
      ForEach(this.values, (value: number) => {
        ListItem() {
          ComputedValueRow({ value: total })
        }
      }, (value: number) => value.toString())
    }
  }
}
