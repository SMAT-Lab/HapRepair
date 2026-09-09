# @hw-ets-eslint/init-list-component

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_a11c56f27e5decd5`

This rule validates the initial values of the width and height attributes of the List component.

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build(){
    List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
      LazyForEach(this.listValue, (item) => {
        ListItem() {
          ListItemComponent({ item: item, listData: listData })
        }
        .editable(true)
      }, item => item)
        .cachedCount(3)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build(){
    List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
      LazyForEach(this.listValue, (item) => {
        ListItem() {
          ListItemComponent({ item: item, listData: listData })
        }
        .editable(true)
      }, item => item)
        .cachedCount(3)
    }
    .height(100)
    .width(100)
  }
}
```

### Rationale

使用List的时候，需要指明height和width
