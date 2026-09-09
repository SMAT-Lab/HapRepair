# @performance/avoid-overusing-custom-component-check

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_4645252f37ab95d0`

During application development, reducing the use of custom components—especially within loops—can exponentially decrease the number of **CustomNode** instances in the **FrameNode** tree, significantly improving page loading and rendering performance. When custom components are necessary, consider using @Builder functions as alternatives. Unlike custom components, @Builder functions do not create new tree nodes in the backend **FrameNode** tree. For example, when displaying card lists using **ForEach**, if the cards only need to present information without requiring complex custom component capabilities like lifecycle functions, you can create an @Builder function instead of a custom card component.

### Triggering pattern

```arkts
import { util } from '@kit.ArkTS';

interface User {
  id: string;
  name: string;
  age?: number;
  avatarImage?: ResourceStr;
  //introduction: string;
  // ...
}

// Create data.
const DEFAULT_BACKGROUND_COLOR = Color.Pink;
const getUsers = () => {
  const USERS: User[] = [{
    id: '1',
    name: 'John Doe',
  }, {
    id: '2',
    name: 'Jane Smith',
  }, {
    id: '3',
    name: 'Alex Brown',
  }];
  return Array.from(Array(30), (item: User, i: number) => {
    return {
      id: util.generateRandomUUID(),
      name: USERS[i%3].name,
      avatarImage: $r('app.media.avatar'),
      age: 18 + i
    } as User;
  });
}

// User card list component.
@Component export struct UserCardList {
  @State users: User[] = getUsers();


  build() {
    List({space: 8}) {
      ForEach(this.users, (item: User) => {
        ListItem() {
          UserCard({name: item.name, age: item.age, avatarImage: item.avatarImage})
        }
      }, (item: User) => item.id)
    }
    .alignListItem(ListItemAlign.Center)
  }
}

// Component for user card customization.
@Component
struct UserCard {
  @Prop avatarImage: ResourceStr;
  @Prop name: string;
  @Prop age: number;

  build() {
    Row() {
      Row(){
        Image(this.avatarImage)
          .size({width: 50, height: 50})
          .borderRadius(25)
          .margin(8)
        Text(this.name)
          .fontSize(30)
      }
      Text(`Age: ${this.age.toString()}`)
        .fontSize(20)
    }
    .backgroundColor(DEFAULT_BACKGROUND_COLOR)
    .justifyContent(FlexAlign.SpaceBetween)
    .borderRadius(8)
    .padding(8)
    .height(66)
    .width('80%')
  }
}
```

### Repair pattern

```arkts
// 1. Create a custom @Builder function component.
@Builder
function UserCardBuilder(name: string, age?: number, avatarImage?: ResourceStr) {
  Row() {
    Row(){
      Image(avatarImage)
        .size({width: 50, height: 50})
        .borderRadius(25)
        .margin(8)
      Text(name)
        .fontSize(30)
    }
    Text(`Age: ${age?.toString()}`)
      .fontSize(20)
  }
  .backgroundColor(DEFAULT_BACKGROUND_COLOR)
  .justifyContent(FlexAlign.SpaceBetween)
  .borderRadius(8)
  .padding(8)
  .height(66)
  .width('80%')
}

@Component
export struct UserCardList {
  @State users: User[] = getUsers();

  aboutToAppear(): void {
    let message = 'hello world';
  }

  build() {
    List({space: 8}) {
      ForEach(this.users, (item: User) => {
        ListItem() {
          // 2. Use the @Builder function in the build function.
          UserCardBuilder(item.name,item.age,item.avatarImage)
        }
      }, (item: User) => item.id)
    }
    .alignListItem(ListItemAlign.Center)
  }
}
```
