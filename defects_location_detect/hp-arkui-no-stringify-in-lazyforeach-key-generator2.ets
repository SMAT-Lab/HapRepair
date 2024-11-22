import { ProductDataSource } from '../m/ProductDataSource';

@Reusable
@Component
struct ProductComponent {
  @State productName: string = '';
  @State productPrice: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.productName = params.productName as string;
    this.productPrice = params.productPrice as number;
  }

  build() {
    Column() {
      Text('Product: ' + this.productName)
        .fontSize(20)
        .fontWeight(20)
      Text('Price: ' + this.productPrice)
        .fontSize(18)
        .fontWeight(20)
    }
  }
}

class ProductItem {
  id: string = '';
  name: string = '';
  price: number = 0;
  description: string = '';
  category: string = '';
}

@Entry
@Component
struct ProductListComponent {
  private dataSource: ProductDataSource = new ProductDataSource();

  aboutToAppear(): void {
    // Generate 1000 products with some data
    for (let index = 0; index < 1000; index++) {
      let item = new ProductItem();
      item.id = index.toString();
      item.name = `Product ${index}`;
      item.price = (index + 1) * 10;
      item.description = `This is the description of product ${index}`;
      item.category = index % 2 === 0 ? 'Electronics' : 'Clothing';
      this.dataSource.pushData(item);
    }
  }

  build() {
    Column() {
      Text('Product List')
        .fontSize(18)
        .fontWeight(24)
      List() {
        LazyForEach(this.dataSource, (item: ProductItem) => {
          ListItem() {
            ProductComponent({ productName: item.name, productPrice: item.price })
          }
          .width('100%')
          .height('20%')
          .border({ width: 1 })
          .borderStyle(BorderStyle.Solid)
        }, (item: ProductItem) => JSON.stringify(item))  // Key generator using JSON.stringify(item)
      }
      .cachedCount(3)
      .height('100%')
      .width('100%')
    }
  }
}
