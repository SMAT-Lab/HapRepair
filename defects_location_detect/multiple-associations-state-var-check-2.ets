import { Product } from '../HomeRepairWithoutDiff/hp-arkts-no-use-any-export-current';

interface Product_interface {
  name: string;
  price: number;
}
@Observed
class CartData {
  totalAmount: number = 0;
  itemCount: number = 0;
}

@Entry
@Component
struct ShoppingCartComponent {
  @State cartData: CartData = new CartData();
  private products: Product_interface[] = [];

  aboutToAppear(): void {
    this.products = [
      { name: 'Apple', price: 1.5 },
      { name: 'Banana', price: 1.0 },
      { name: 'Orange', price: 2.0 },
    ];
  }

  build() {
    Column() {
      ForEach(this.products, (product: Product_interface, index) => {
        ProductItemComponent({ product: product, index: index, sharedCart: this.cartData });
      }, (product: Product_interface) => product.name)

      Button('Add Apple to Cart')
        .onClick(() => {
          this.cartData.itemCount += 1;
          this.cartData.totalAmount += this.products[0].price;
        })
        .padding(10)

      Button('Clear Cart')
        .onClick(() => {
          this.cartData.itemCount = 0;
          this.cartData.totalAmount = 0;
        })
        .padding(10)
    }
  }
}


@Component
struct ProductItemComponent {
  @Prop product: Product_interface;
  @Prop index: number;
  @Link sharedCart: CartData;

  build() {
    Row() {
      Text(`Product: ${this.product.name}, Price: $${this.product.price.toFixed(2)}`)
        .fontSize(14)
        .padding(5)

      Text(`Cart Total: $${this.sharedCart.totalAmount.toFixed(2)}, Items: ${this.sharedCart.itemCount}`)
        .fontSize(14)
        .padding(5)
    }
  }
}
