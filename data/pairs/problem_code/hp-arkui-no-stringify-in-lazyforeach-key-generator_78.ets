import { MyDataSource } from './MyDataSource';
// 此处为复用的自定义组件
@Reusable
@Component
struct ChildComponent {
  @State desc: string = '';
  @State sum: number = 0;
  @State avg: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.desc = params.desc as string;
    this.sum = params.sum as number;
    this.avg = params.avg as number;
  }

  build() {
    Column() {
      Text('子组件' + this.desc)
        .fontSize(30)
        .fontWeight(30)
      Text('结果' + this.sum)
        .fontSize(30)
        .fontWeight(30)
      Text('平均值' + this.avg)
        .fontSize(30)
        .fontWeight(30)
    }
  }
}

class Item {
  advertInfos: Model[] = []
  productPrice: PriceInfo[] = []
  addresses: string[] = []
  id: string = ''
}

class Model {
  pictureUrl: string = ""
  name: string = ""
  comments: string = ""
  desc: string = ""
  linkParam: string = ""
  mcInfo: string = ""
  label: string = ""
  cgType: string = ""

  constructor(pictureUrl: string, name: string, comments: string, desc: string, linkParam: string, mcInfo: string,
    label: string, cgType: string) {
    this.pictureUrl = pictureUrl;
    this.name = name;
    this.comments = comments;
    this.desc = desc;
    this.linkParam = linkParam;
    this.mcInfo = mcInfo;
    this.label = label;
    this.cgType = cgType;
  }
}

class PriceInfo {
  price: number = 0;
  level: number = 1;

  constructor(price: number, level: number) {
    this.price = price;
    this.level = level;
  }
}

@Entry
@Component
struct MyComponent {
  private data: MyDataSource = new MyDataSource();

  aboutToAppear(): void {
    for (let index = 0; index < 20; index++) {
      let item = new Item()
      for (let i = 0; i < 1000; i++) {
        item.advertInfos.push(new Model("Product A", "Product A", "Product A", "Product A", "Product A", "Product A", "Product A", "Product A"));
        item.productPrice.push(new PriceInfo(1.99, 123456));
        item.addresses.push("Beijing")
      }
      item.id = index.toString();
      this.data.pushData(item.productPrice[0].price)
    }
  }

  build() {
    Column() {
      Text('Use the time-consuming function `JSON.stringify (item)` to generate a key')
        .fontSize(12)
        .height('16')
        .margin({
          top: 5,
          bottom: 10
        })
      List() {
        LazyForEach(this.data, (item: Item) => {
          ListItem() {
            ChildComponent({ desc: item.id, sum: 0, avg: 0 })
          }
          .width('100%')
          .height('10%')
          .border({ width: 1 })
          .borderStyle(BorderStyle.Dashed)
        }, (item: Item) => JSON.stringify(item))
      }
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}