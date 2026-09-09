@Entry
@Component
struct CheckoutTotals {
  @State payable: number = 0;

  recalculate(subtotal: number, deliveryFee: number, discount: number): void {
    let nextPayable: number = subtotal;
    nextPayable += deliveryFee;
    nextPayable -= discount;
    this.payable = nextPayable;
  }

  build() {
    Row({ space: 12 }) {
      Text(`Amount due: ${this.payable}`)
      Button('Recalculate')
        .onClick(() => this.recalculate(86, 8, 5))
    }
  }
}
