# @performance/hp-arkui-use-reusable-component

Static repair references: 6. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_2271edeace47c78d`

建议复杂组件的定义，尽量使用组件复用。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.data, item => {
        GridItem() {
          Column() {
            Image(this.productData[item % 14].uri)
              .width('100%')
              .height('40%')
              .objectFit(ImageFit.Contain)
              .margin({ bottom: 40 })
            Text(this.productData[item % 14].title)
              .fontSize(16 * this.ratio)
              .fontWeight(600)
              .margin({ bottom: 10, left: 10 })
            Row() {
              Text(this.productData[item % 14].labels)
                .fontSize(10 * this.ratio)
                .border({ width: 1, color: '#FA808080' })
                .margin({ bottom: 2, left: 10 })
                .padding(2)
            }
            .margin({ bottom: 2 })

            Text(this.labels)
              .fontSize(16 * this.ratio)
              .fontColor(Color.Red)
              .margin({ left: 10 })
          }
          .alignItems(HorizontalAlign.Start)
        }
        .width('95%')
        .height(300)
        .border({ width: 1, color: '#70808080', radius: 10 })
        .margin({ top: 3, bottom: 3 })
        .backgroundColor(Color.White)
      }, item => item.toString())
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App{
  build(){
    Column(){
      LazyForEach(this.data, item => {
          GridItem() {
            ProductGridItemComponent({
              uri: this.productData[item % 14].uri,
              title: this.productData[item % 14].title,
              labels: this.productData[item % 14].labels,
              price: item + 1,
              ratio: this.ratio // 你可能需要传递这个比例计算
            })
          }
          .width('95%')
          .height(300)
          .border({ width: 1, color: '#70808080', radius: 10 })
          .margin({ top: 3, bottom: 3 })
          .backgroundColor(Color.White)
          .padding({ bottom: 60 })
        }, item => item.toString())
    }

    .width('100%')
    .backgroundColor('#10000000')
  }
}

@Reusable
@Component
struct ProductGridItemComponent {
  @State uri: string = "";
  @State title: string = "";
  @State labels: string = "";
  @State price: number = 0;
  @State ratio: number = 0;

  constructor(uri: string, title: string, labels: string, price: number, ratio: number) {
    super()
    this.uri = uri;
    this.title = title;
    this.labels = labels;
    this.price = price;
    this.ratio = ratio;
  }

  build() {
    Column() {
      Image(this.uri)
        .width('100%')
        .height('40%')
        .objectFit(ImageFit.Contain)
        .margin({ bottom: 40 });

      Text(this.title)
        .fontSize(16 * this.ratio)
        .fontWeight(600)
        .margin({ bottom: 10, left: 10 });

      Row() {
        Text(this.labels)
          .fontSize(10 * this.ratio)
          .border({ width: 1, color: '#FA808080' })
          .margin({ bottom: 2, left: 10 })
          .padding(2);
      }
      .margin({ bottom: 2 });

      Text(`￥${this.price}`)
        .fontSize(16 * this.ratio)
        .fontColor(Color.Red)
        .margin({ left: 10 });
    }
    .alignItems(HorizontalAlign.Start);
  }
}
```

### Rationale

LazyForEach里反复使用大量的代码来表示某个组件，封装起来来进行组件复用

## Example 2: `pair_41ffb5f50ae2fe48`

建议复杂组件的定义，尽量使用组件复用。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  layoutOption: GridLayoutOptions =  {
    regularSize: [1, 1]
  }
  build() {
    Grid(this.scroller, this.layoutOption) {
      LazyForEach(this.albumsDataSource, (item: LazyItem<AlbumDataItem>): void => {
        if ((item != undefined && item != null) && (item.get() != undefined && item.get() != null)) {
          if (item.index === 0) {
            GridItem() {
              AlbumSelectGridItemNewStyle({
                item: item.get(),
                isBigCard: true,
              })
            }.columnStart(0).columnEnd(1)
          } else if (item != null) {
            GridItem() {
              AlbumSelectGridItemNewStyle({
                item: item.get(),
                isBigCard: false,
              })
            }
          }
        }
      }, (item: LazyItem<AlbumDataItem>): string => (item != undefined && item != null) &&
        (item.get() != undefined && item.get() != null) ?
      item.getHashCode() : item.id)
    }
    .cachedCount(2)
  }
}
```

### Repair pattern

```arkts
@Reusable
@Component
struct AlbumGridItemComponent {
  @State item: AlbumDataItem = new AlbumDataItem("1", 0, "", 0, "", 0, 0)
  isBigCard: boolean = false

  build() {
    GridItem() {
      AlbumSelectGridItemNewStyle({
        item: this.item,
        isBigCard: this.isBigCard,
      })
    }
  }
}

@Reusable
@Component
struct AlbumSelectGridItemNewStyle {
  @Prop @Watch("aboutToReuse") item: AlbumDataItem = new AlbumDataItem()
  @Prop @Watch("aboutToReuse") isBigCard: boolean = false

  aboutToReuse(params: any): void {

  }
  build() {
    Grid(this.scroller) {
      LazyForEach(this.albumsDataSource, (item: LazyItem<AlbumDataItem>): void => {
        if (item && item.get()) {

          if (item.index === 0) {
            // 处理第一个专辑项
            AlbumGridItemComponent({
              item: item.get(),
              isBigCard: true,
            }); // 设置网格的起始和结束列
          } else {
            // 处理普通的专辑项
            AlbumGridItemComponent({
              item: item.get(),
              isBigCard: false,
            });
          }
        }
      }, (item: LazyItem<AlbumDataItem>): string => (item != undefined && item != null) &&
        (item.get() != undefined && item.get() != null) ?
      item.getHashCode() : item.id)

    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
```

### Rationale

Grid()里反复使用大量的代码来表示某个组件，建议封装组件并进行复用

## Example 3: `pair_44f64b90a16f964d`

建议复杂组件的定义，尽量使用组件复用。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
      LazyForEach(this.listValue, (item) => {
        ListItem() {
          Column() {
            Text('     ' + item)
              .height(30)
              .fontSize(16)
              .width('100%')
              .textAlign(TextAlign.Start)
            Column() {
              ForEach(listData, (ele) => {
                if (ele.substr(0, 1) == item.toUpperCase() || ele.substr(0, 1) == item.toLowerCase()) {
                  Text('' + ele)
                    .fontSize(16)
                    .width('100%')
                    .textAlign(TextAlign.Center)
                    .margin({ top: 5, bottom: 5 })
                    .height(72)
                    .borderRadius(24)
                    .backgroundColor('#ffffff')
                }
              }, (item: string)=>item)
            }
          }
        }
        .editable(true)
      }, item => item)
    }
    .cachedCount(2)
    .height("100%")
    .width("100%")
  }
}
```

### Repair pattern

```arkts
@Reusable
@Component
struct ListItemComponent {
  item: string = ''
  @State listData: string[] = []

  build() {
    Column() {
      Text('     ' + this.item)
        .height(30)
        .fontSize(16)
        .width('100%')
        .textAlign(TextAlign.Start)

      // 这里使用一个内部 ForEach 遍历 listData
      ForEach(this.listData, (ele: string) => {
        if (ele.substr(0, 1) === this.item.toUpperCase() || ele.substr(0, 1) === this.item.toLowerCase()) {
          Text('' + ele)
            .fontSize(16)
            .width('100%')
            .textAlign(TextAlign.Center)
            .margin({ top: 5, bottom: 5 })
            .height(72)
            .borderRadius(24)
            .backgroundColor('#ffffff')
        }
      }, (item: string)=>item)
    }
  }
}

class FruitValue implements IDataSource {
  public fruit: string[] = ["A", "B", "C", "D"]
  totalCount(): number {
    return this.fruit.length
  }

  getData(index: number): string {
    return this.fruit[index]
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error('Method not implemented.')
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error('Method not implemented.')
  }

}

@Entry
@Component
struct MyComponent {
  private listValue: FruitValue = new FruitValue()
  private listData: string[] = ["Apple", "Avocado", "Banana", "Blueberry", "Cherry", "Date"]

  build() {
    List() {
      LazyForEach(this.listValue, (item: string) => {
        ListItem() {
          // 使用 reusable component 传递 item 和 listData
          ListItemComponent({
            item: item,
            listData: this.listData
          })
        }
      }, (item: string) => item)
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
```

### Rationale

List()里反复使用大量的代码来表示某个组件，建议封装组件并进行复用

## Example 4: `pair_8ac010bdf4ee51e1`

建议复杂组件的定义，尽量使用组件复用。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    LazyForEach(this.fileList, (listItem: fileAccess.FileInfo) => {
      ListItem() {
        FileListItemComponent({
          fileListItem: listItem,
          itemFileList: $fileList,
          itemShowDeleteButton: $showDeleteButton,
          itemIsNoFile: $isNoFile
        })
      }
    }, (listItem: fileAccess.FileInfo) => listItem.fileName)
  }
}

@Component
struct FileListItemComponent {
  @Link itemFileList: DocumentDataSource;
  @Link itemShowDeleteButton: boolean;
  @Link itemIsNoFile: boolean;
  private fileListItem: fileAccess.FileInfo = {} as fileAccess.FileInfo;
  private itemClickFunction: (fileAsset: fileAccess.FileInfo) => void = () => {
    prompt.showToast({
      duration: ONE_SECOND,
      message: $r('app.string.not_supported_tip')
    })
  }

  @Styles
  itemPressedStyles() {
    .backgroundColor($r('app.color.item_pressed'))
    .borderRadius(10)
  }

  build() {
      Row() {
        Image($r('app.media.default_document'))
          .objectFit(ImageFit.Fill)
          .width(40)
          .height(40)
          .margin({ left: 20 })

        Column() {
          Text(this.fileListItem.fileName)
            .maxLines(1)
            .width('75%')
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .fontSize(16)
            .fontColor($r('app.color.black'))

          Text(this.fileListItem.mtime + ' - ' + this.fileListItem.size + 'B')
            .fontSize(12)
            .margin({ top: 5 })
            .fontColor($r('app.color.font_gray'))
        }
        .margin({ left: 10 })
        .alignItems(HorizontalAlign.Start)
      }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.fileList, (listItem: fileAccess.FileInfo) => {
        ListItem() {
          FileListItemComponent({
            fileListItem: listItem,
            itemFileList: $fileList,
            itemShowDeleteButton: $showDeleteButton,
            itemIsNoFile: $isNoFile
          })
        }
      }, (listItem: fileAccess.FileInfo) => listItem.fileName)
    }
  }
}

@Reuse
@Component
struct FileListItemComponent {
  @Link itemFileList: DocumentDataSource;
  @Link itemShowDeleteButton: boolean;
  @Link itemIsNoFile: boolean;
  private fileListItem: fileAccess.FileInfo = {} as fileAccess.FileInfo;
  private itemClickFunction: (fileAsset: fileAccess.FileInfo) => void = () => {
    prompt.showToast({
      duration: ONE_SECOND,
      message: $r('app.string.not_supported_tip')
    })
  }

  @Styles
  itemPressedStyles() {
    .backgroundColor($r('app.color.item_pressed'))
    .borderRadius(10)
  }

  build() {
      Row() {
        Image($r('app.media.default_document'))
          .objectFit(ImageFit.Fill)
          .width(40)
          .height(40)
          .margin({ left: 20 })

        Column() {
          Text(this.fileListItem.fileName)
            .maxLines(1)
            .width('75%')
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .fontSize(16)
            .fontColor($r('app.color.black'))

          Text(this.fileListItem.mtime + ' - ' + this.fileListItem.size + 'B')
            .fontSize(12)
            .margin({ top: 5 })
            .fontColor($r('app.color.font_gray'))
        }
        .margin({ left: 10 })
        .alignItems(HorizontalAlign.Start)
      }
  }
}
```

### Rationale

LazyForEach里反复使用大量的代码来表示某个组件，给该组件添加@Reusable注解来表示该组件可复用

## Example 5: `pair_9c554fd57e18f81e`

建议复杂组件的定义，尽量使用组件复用。

### Triggering pattern

```arkts
import { MyDataSource } from './MyDataSource';
import { GoodItems } from './data/DataEntry';

@Entry
@Component
struct MyComponent{
  private data: MyDataSource = new MyDataSource();

  build() {
    Column() {
      LazyForEach(this.data, (item: GoodItems) => {
        GridItem() {
          Column() {
            Text(item.introduce)
              .fontSize(14)
              .padding({ left: 5, right: 5 })
              .margin({ top: 5 })
            Row() {
              Text('￥')
                .fontSize(10)
                .fontColor(Color.Red)
                .baselineOffset(-4)
              Text(item.price)
                .fontSize(16)
                .fontColor(Color.Red)
              Text(item.numb)
                .fontSize(10)
                .fontColor(Color.Gray)
                .baselineOffset(-4)
                .margin({ left: 5 })

            }
            .width('100%')
            .justifyContent(FlexAlign.SpaceBetween)
            .padding({ left: 5, right: 5 })
            .margin({ top: 15 })
          }
          .borderRadius(10)
          .backgroundColor(Color.White)
          .clip(true)
          .width('100%')
          .height(290)
        }
      }, (item: GoodItems) => item.index)
    }
  }
}
```

### Repair pattern

```arkts
import { MyDataSource } from './MyDataSource';
import { GoodItems } from './data/DataEntry';

@Reusable
@Component
struct GoodItemComponent {
  @State introduce: string = ''
  @State price: string = ''
  @State numb: string = ''

  aboutToReuse(params: Record<string, ESObject>) {
    this.introduce = params.introduce
    this.price = params.price
    this.numb = params.numb
  }

  build() {
    Column() {
      Text(this.introduce)
        .fontSize(14)
        .padding({ left: 5, right: 5 })
        .margin({ top: 5 })
      Row() {
        Text('￥')
          .fontSize(10)
          .fontColor(Color.Red)
          .baselineOffset(-4)
        Text(this.price)
          .fontSize(16)
          .fontColor(Color.Red)
        Text(this.numb)
          .fontSize(10)
          .fontColor(Color.Gray)
          .baselineOffset(-4)
          .margin({ left: 5 })

      }
      .width('100%')
      .justifyContent(FlexAlign.SpaceBetween)
      .padding({ left: 5, right: 5 })
      .margin({ top: 15 })
    }
  }
}

@Entry
@Component
struct MyComponent{
  private data: MyDataSource = new MyDataSource();

  build() {
    Column() {
      LazyForEach(this.data, (item: GoodItems, index) => {
        GridItem() {
          GoodItemComponent({
            introduce: item.introduce,
            price: item.price,
            numb: item.numb,
          }).reuseId(item.numb)
        }
      }, (item: GoodItems) => item.index)
    }
  }
}
```

### Rationale

GridItem()里反复使用大量的代码来表示某个组件，建议封装组件并进行复用

## Example 6: `pair_b4cc51e6b7cc7c61`

建议复杂组件的定义，尽量使用组件复用。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  layoutOptions: GridLayoutOptions = {
    regularSize: [1,1]
  }
  build() {
    Grid(this.scroller, this.layoutOptions) {
      LazyForEach(this.groupDataSource, (item: LazyItem<UserFileDataItem>): void => {
        GridItem() {
          ImageGridItemComponent({
            lazyItem: item,
            mediaItem: item.get(),
            pageName: Constants.PHOTO_TRANSITION_ALBUM,
            isSelectUpperLimited: $isSelectUpperLimited
          })
        }
        .aspectRatio(1)
        .columnStart(item.get().index % this.gridRowCount)
        .columnEnd(item.get().index % this.gridRowCount)
      }, (item: LazyItem<AlbumDataItem>): string => item.getHashCode())
    }
    .cachedCount(2)
  }
}
```

### Repair pattern

```arkts
@Reusable
@Component
struct ImageGridItem {
  @State lazyItem: LazyItem<UserFileDataItem> = new LazyItem<UserFileDataItem>(
    new UserFileDataItem("initial_selection", ["arg1", "arg2"], "device123", 0), // 创建 UserFileDataItem 实例
    0
  );
  @State mediaItem: UserFileDataItem | undefined = undefined; // 初始化为 undefined
  @State pageName: string = Constants.PHOTO_TRANSITION_ALBUM; // 直接使用常量初始化
  @State isSelectUpperLimited: boolean = false; // 设置初始值为 false

  constructor(lazyItem: LazyItem<UserFileDataItem>, pageName?: string, isSelectUpperLimited?: boolean) {
    super()
    // 使用 defaults 处理可选参数
    this.lazyItem = lazyItem;
    this.mediaItem = lazyItem.item; // 从 lazyItem 初始化 mediaItem
    // 如果提供了，才使用来自参数的 pageName 和 isSelectUpperLimited
    if (pageName) {
      this.pageName = pageName;
    }
    if (isSelectUpperLimited !== undefined) {
      this.isSelectUpperLimited = isSelectUpperLimited;
    }
  }

  build() {
    if (this.lazyItem && this.mediaItem) {
      ImageGridItemComponent({
        lazyItem: this.lazyItem,
        mediaItem: this.mediaItem,
        pageName: Constants.PHOTO_TRANSITION_ALBUM,
        isSelectUpperLimited: $isSelectUpperLimited
      })
    }
  }
}
@Reusable
@Component
struct ImageGridItemComponent {
  @State lazyItem: LazyItem<UserFileDataItem> = new LazyItem<UserFileDataItem>()
  @State mediaItem: UserFileDataItem | undefined = undefined;
  @State pageName: string = ""; // 直接使用常量初始化
  @State isSelectUpperLimited: boolean = false; // 设置初始值为 false

  build() {
    Grid(this.scroller) {
      LazyForEach(this.groupDataSource, (item: LazyItem<UserFileDataItem>): void => {
        GridItem() {
          ImageGridItem()
        }
        .aspectRatio(1)
        .columnStart(item.get().index % this.gridRowCount)
        .columnEnd(item.get().index % this.gridRowCount)
      }, (item: LazyItem<AlbumDataItem>): string => item.getHashCode())
    }
    .columnsTemplate('1fr '.repeat(this.gridRowCount))
    .columnsGap(Constants.GRID_GUTTER)
    .rowsGap(Constants.GRID_GUTTER)
    .cachedCount(Constants.GRID_CACHE_ROW_COUNT)

    GridScrollBar({ scroller: this.scroller, isHideScrollBar: $isHideScrollBar });
  }
}
```

### Rationale

Grid()里反复使用大量的代码来表示某个组件，建议封装组件并进行复用
