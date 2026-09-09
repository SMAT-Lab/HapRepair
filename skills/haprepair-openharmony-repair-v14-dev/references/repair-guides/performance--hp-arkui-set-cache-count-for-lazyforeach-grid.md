# @performance/hp-arkui-set-cache-count-for-lazyforeach-grid

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_36fbc6627ac4d575`

建议在Grid下使用LazyForEach时设置合理的cacheCount

### Triggering pattern

```arkts
// 文件：ImageGallery.ets
@Entry
@Component
struct ImageGallery {
  private imageList: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 500; i++) {
      this.imageList.push('image_' + i + '.png');
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.imageList, (image: string) => {
          GridItem() {
            Image($r('app.media.' + image))
              .width(100)
              .height(100)
          }
        }, (image: string) => image)
      }
      // 未设置缓存数量
      .columnsTemplate('1fr 1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height('100%')
    }
  }
}
```

### Repair pattern

```arkts
// 文件：ImageGallery.ets
@Entry
@Component
struct ImageGallery {
  private imageList: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 500; i++) {
      this.imageList.push('image_' + i + '.png');
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.imageList, (image: string) => {
          GridItem() {
            Image($r('app.media.' + image))
              .width(100)
              .height(100)
          }
        }, (image: string) => image)
      }
      // 设置缓存数量
      .cachedCount(4)
      .columnsTemplate('1fr 1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height('100%')
    }
  }
}
```

### Rationale

由于图片库中包含大量图片，未设置 cachedCount 可能导致滚动时图片加载不及时。通过设置 cachedCount(4)，Grid 会预先缓存 4 个 GridItem，提升图片加载和滚动的性能。

## Example 2: `pair_634c38f9d9c59d94`

建议在Grid下使用LazyForEach时设置合理的cacheCount

### Triggering pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

@Entry
@Component
struct MyComponent {
  // 数据源
  private data: MyDataSource = new MyDataSource();

  aboutToAppear() {
    for (let i = 1; i < 1000; i++) {
      this.data.pushData(i);
    }
  }

  build() {
    Column({ space: 5 }) {
      Grid() {
        LazyForEach(this.data, (item: number) => {
          GridItem() {
            // 使用可复用自定义组件
            // 业务逻辑
          }
        }, (item: string) => item)
      }
      // 未设置GridItem的缓存数量
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .margin(10)
      .height(500)
      .backgroundColor(0xFAEEE0)
    }
  }
}
```

### Repair pattern

```arkts
// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

@Entry
@Component
struct MyComponent {
  // 数据源
  private data: MyDataSource = new MyDataSource();

  aboutToAppear() {
    for (let i = 1; i < 1000; i++) {
      this.data.pushData(i);
    }
  }

  build() {
    Column({ space: 5 }) {
      Grid() {
        LazyForEach(this.data, (item: number) => {
          GridItem() {
            // 使用可复用自定义组件
            // 业务逻辑
          }
        }, (item: string) => item)
      }
      // 设置GridItem的缓存数量
      .cachedCount(2)
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .margin(10)
      .height(500)
      .backgroundColor(0xFAEEE0)
    }
  }
}
```

### Rationale

在Grid下使用LazyForEach时，设置GridItem的缓存数量为2

## Example 3: `pair_7a59713f11ebaeb2`

建议在Grid下使用LazyForEach时设置合理的cacheCount

### Triggering pattern

```arkts
// 文件：VideoGrid.ets
@Entry
@Component
struct VideoGrid {
  private videos: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 600; i++) {
      this.videos.push('video_' + i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.videos, (video: string) => {
          GridItem() {
            // 显示视频缩略图
            Image($r('app.media.' + video + '_thumb'))
              .width(150)
              .height(100)
          }
        }, (video: string) => video)
      }
      // 未设置缓存数量
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(8)
      .rowsGap(8)
      .height(600)
    }
  }
}
```

### Repair pattern

```arkts
// 文件：VideoGrid.ets
@Entry
@Component
struct VideoGrid {
  private videos: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 600; i++) {
      this.videos.push('video_' + i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.videos, (video: string) => {
          GridItem() {
            // 显示视频缩略图
            Image($r('app.media.' + video + '_thumb'))
              .width(150)
              .height(100)
          }
        }, (video: string) => video)
      }
      // 设置缓存数量
      .cachedCount(3)
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(8)
      .rowsGap(8)
      .height(600)
    }
  }
}
```

### Rationale

对于包含大量视频缩略图的网格布局，未设置 cachedCount 可能导致加载延迟和滚动卡顿。通过设置 cachedCount(3)，可以在滚动时预先加载 3 个 GridItem，提升性能和用户体验。

## Example 4: `pair_bbc79637fcc60756`

建议在Grid下使用LazyForEach时设置合理的cacheCount

### Triggering pattern

```arkts
@Entry
@Component
struct ProductGrid {
  private products: number[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 1000; i++) {
      this.products.push(i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.products, (product: number) => {
          GridItem() {
            // 自定义组件，显示产品信息
            Text('产品编号：' + product)
              .fontSize(16)
              .width('100%')
              .height(80)
              .textAlign(TextAlign.Center)
          }
        }, (product: number) => product.toString())
      }
      // 未设置 Grid 的缓存数量
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height(600)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ProductGrid {
  private products: number[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 1000; i++) {
      this.products.push(i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.products, (product: number) => {
          GridItem() {
            // 自定义组件，显示产品信息
            Text('产品编号：' + product)
              .fontSize(16)
              .width('100%')
              .height(80)
              .textAlign(TextAlign.Center)
          }
        }, (product: number) => product.toString())
      }
      // 设置 GridItem 的缓存数量
      .cachedCount(3)
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height(600)
    }
  }
}
```

### Rationale

在原始代码中，Grid 使用 LazyForEach 渲染大量数据项，但未设置 cachedCount。这可能导致在快速滚动时出现性能问题。通过设置 cachedCount(3)，我们指示 Grid 预先缓存 3 个 GridItem，以提高滚动性能和界面流畅度。

## Example 5: `pair_e544f4127c593dec`

建议在Grid下使用LazyForEach时设置合理的cacheCount

### Triggering pattern

```arkts
// 文件：ArticleGrid.ets
@Entry
@Component
struct ArticleGrid {
  private articles: { id: number; title: string }[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 800; i++) {
      this.articles.push({ id: i, title: '文章标题 ' + i });
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.articles, (article) => {
          GridItem() {
            Text(article.title)
              .fontSize(14)
              .width('100%')
              .height(60)
          }
        }, (article) => article.id.toString())
      }
      // 未设置缓存数量
      .columnsTemplate('1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .height(500)
    }
  }
}
```

### Repair pattern

```arkts
// 文件：ArticleGrid.ets
@Entry
@Component
struct ArticleGrid {
  private articles: { id: number; title: string }[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 800; i++) {
      this.articles.push({ id: i, title: '文章标题 ' + i });
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.articles, (article) => {
          GridItem() {
            Text(article.title)
              .fontSize(14)
              .width('100%')
              .height(60)
          }
        }, (article) => article.id.toString())
      }
      // 设置缓存数量
      .cachedCount(2)
      .columnsTemplate('1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .height(500)
    }
  }
}
```

### Rationale

在大量文章标题的网格布局中，未设置 cachedCount 可能导致滚动不流畅。通过设置 cachedCount(2)，可以缓存 2 个 GridItem，提高滚动时的性能。
