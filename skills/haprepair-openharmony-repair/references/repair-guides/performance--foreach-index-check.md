# @performance/foreach-index-check

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1db26b909e15ad35`

The **keyGenerator** parameter in **ForEach** is a callback that allows you to define your own key generation rules. However, if the generated keys incorporate index values, inserting new data into the data source can trigger a cascading effect. This insertion causes all subsequent items to have altered keys, potentially leading to unnecessary reconstruction of numerous components and a decline in rendering performance.

### Triggering pattern

```arkts
@Entry
@Component
struct ForeachTest {
  private data: string[] = ['one', 'two', 'three'];

  build() {
    RelativeContainer() {
      List() {
        // warning line
        ForEach(this.data, (item: string, index: number) => {
          ListItem() {
            Text(item);
          }
        }, (item: string, index: number) => item + index)
      }
      .width('100%')
      .height('100%')
    }
    .height('100%')
    .width('100%')
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ForeachTest {
  private data: string[] = ['one', 'two', 'three'];

  build() {
    RelativeContainer() {
      List() {
        ForEach(this.data, (item: string, index: number) => {
          ListItem() {
            Text(item);
          }
        }, (item: string, index: number) => item)
      }
      .width('100%')
      .height('100%')
    }
    .height('100%')
    .width('100%')
  }
}
```
