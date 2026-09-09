@Entry
@Component
struct CheckoutTotals {
  @State payable: number = 0;

  recalculate(subtotal: number, deliveryFee: number, discount: number): void {
    this.payable = subtotal;
    this.payable += deliveryFee;
    this.payable -= discount;
  }

  build() {
    Row({ space: 12 }) {
      Text(`Amount due: ${this.payable}`)
      Button('Recalculate')
        .onClick(() => this.recalculate(86, 8, 5))
    }
  }
}
