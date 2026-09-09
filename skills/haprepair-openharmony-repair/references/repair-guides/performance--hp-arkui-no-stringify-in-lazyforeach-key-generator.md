# @performance/hp-arkui-no-stringify-in-lazyforeach-key-generator

Static repair references: 7. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_29100124683ad15f`

在使用LazyForEach进行组件复用的key生成器函数里，不要使用stringify。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.taskList, (task: Task) => {
        ListItem() {
          TaskComponent({ title: task.title });
        }
      }, (task: Task) => JSON.stringify(task));
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
      LazyForEach(this.taskList, (task: Task) => {
        ListItem() {
          TaskComponent({ title: task.title });
        }
      }, (task: Task) => task.taskId.toString());
    }
  }
}
```

### Rationale

序列化整个任务对象可能导致不必要的性能负担，并且在更新任务对象时可能会出现键冲突。

## Example 2: `pair_2f317172477c30da`

在使用LazyForEach进行组件复用的key生成器函数里，不要使用stringify。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Grid(this.scroller) {
      LazyForEach(this.albumsDataSource, (item: LazyItem<AlbumDataItem>): void => {
        if (item && item.get()) {
          Text()
        }
      }, (item: LazyItem<AlbumDataItem>): string => (item != undefined && item != null) &&
        (item.get() != undefined && item.get() != null) ?
      item.getHashCode() : JSON.stringify(item))
    }
    .cachedCount(2)
    .width("100%")
    .height('100%')
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Grid(this.scroller) {
      LazyForEach(this.albumsDataSource, (item: LazyItem<AlbumDataItem>): void => {
        if (item && item.get()) {
          Text()
        }
      }, (item: LazyItem<AlbumDataItem>): string => (item != undefined && item != null) &&
        (item.get() != undefined && item.get() != null) ?
      item.getHashCode() : item.item.id)
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
```

### Rationale

组件复用的key生成器函数里，不要使用stringify，这里item是一个LazyItem, item.item是一个AlbumDataItem, 而item.item.index是对应的唯一的键

## Example 3: `pair_556aa054706b8244`

在使用LazyForEach进行组件复用的key生成器函数里，不要使用stringify。

### Triggering pattern

```arkts
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
```

### Repair pattern

```arkts
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
        }, (item: Item) => item.id.toString())
      }
      .height('100%')
      .width('100%')
      .cachedCount(2)
    }
  }
}
```

### Rationale

组件复用的key生成器函数里，不要使用stringify，这里改成使用item的id来表示key

## Example 4: `pair_55f812df6ed03204`

在使用LazyForEach进行组件复用的key生成器函数里，不要使用stringify。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.userList, (user: User) => {
        ListItem() {
          UserComponent({ name: user.name, age: user.age });
        }
      }, (user: User) => JSON.stringify(user));
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
      LazyForEach(this.userList, (user: User) => {
        ListItem() {
          UserComponent({ name: user.name, age: user.age });
        }
      }, (user: User) => user.id.toString());
    }
  }
}
```

### Rationale

使用 JSON.stringify(user) 生成 key 时，可能导致不必要的性能消耗，并且一旦对象状态发生变化，字符串化结果可能会不同，引发意外的重新渲染。

## Example 5: `pair_6fa581d4ed749bc5`

在使用LazyForEach进行组件复用的key生成器函数里，不要使用stringify。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.fileList, (listItem: userFileManager.FileAsset) => {
        ListItem() {
          FileListItemComponent({
            itemFileList: $fileList,
            fileListItem: listItem,
            itemClickFunction: this.itemClickFunction,
            uri: listItem.uri,
            itemShowDeleteButton: $showDeleteButton,
            itemIsNoFile: $isNoFile
          })
        }
      }, (listItem: userFileManager.FileAsset) => JSON.stringify(listItem.displayName))
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
    LazyForEach(this.fileList, (listItem: userFileManager.FileAsset) => {
      ListItem() {
        FileListItemComponent({
          itemFileList: $fileList,
          fileListItem: listItem,
          itemClickFunction: this.itemClickFunction,
          uri: listItem.uri,
          itemShowDeleteButton: $showDeleteButton,
          itemIsNoFile: $isNoFile
        })
      }
    }, (listItem: userFileManager.FileAsset) => listItem.displayName.name)
  }
}
```

### Rationale

组件复用的key生成器函数里，不要使用stringify，这里改成使用listItem.displayName.name来表示key

## Example 6: `pair_7a7f65c7674e52ad`

在使用LazyForEach进行组件复用的key生成器函数里，不要使用stringify。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.productList, (product: Product) => {
        ListItem() {
          ProductComponent({ productName: product.name });
        }
      }, (product: Product) => JSON.stringify({ id: product.id, name: product.name }));
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
      LazyForEach(this.productList, (product: Product) => {
        ListItem() {
          ProductComponent({ productName: product.name });
        }
      }, (product: Product) => product.id.toString());
    }
  }
}
```

### Rationale

在 JSON.stringify 中包含多个属性会导致性能开销，并且可能不必要地添加复杂性。

## Example 7: `pair_a90e7a951207e97d`

在使用LazyForEach进行组件复用的key生成器函数里，不要使用stringify。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.notificationList, (notification: Notification) => {
        ListItem() {
          NotificationComponent({ message: notification.message });
        }
      }, (notification: Notification) => JSON.stringify(notification.details));
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
      LazyForEach(this.notificationList, (notification: Notification) => {
        ListItem() {
          NotificationComponent({ message: notification.message });
        }
      }, (notification: Notification) => notification.id.toString());
    }
  }
}
```

### Rationale

JSON.stringify(notification.details) 可能包含大量信息，并且在不同状态下键的生成会有所不同，导致不必要的复杂性和性能浪费。
