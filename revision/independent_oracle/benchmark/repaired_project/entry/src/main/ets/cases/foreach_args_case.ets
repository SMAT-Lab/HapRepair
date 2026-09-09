interface InventoryRow {
  sku: string;
  quantity: number;
}

@Component
struct InventoryListCase {
  private rows: InventoryRow[] = [
    { sku: 'A-17', quantity: 4 },
    { sku: 'B-08', quantity: 9 }
  ];

  build() {
    List() {
      ForEach(this.rows, (row: InventoryRow) => {
        ListItem() {
          Row() {
            Text(row.sku)
            Text(`${row.quantity}`)
          }
        }
      }, (row: InventoryRow) => row.sku)
    }
    .width('100%')
    .height(320)
  }
}
