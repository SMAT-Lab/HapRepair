# @performance/hp-arkui-remove-unchanged-state-var

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0581ce92952b7e10`

建议移除未改变的状态变量设置。

通用丢帧场景下，建议优先修改。

### Triggering pattern

```arkts
class Position {
  x: number = 0;
  y: number = 0;
}

@Component
struct MyComponent {
  @State position: Position = new Position();
  @State title: string = "Welcome";

  build() {
    Column() {
      Text(this.title)
        .fontSize(24)
      Button("Move")
        .onClick(() => {
          this.position.x += 10;
          this.position.y += 10;
        })
    }
    .position({
      x: this.position.x,
      y: this.position.y
    })
  }
}
```

### Repair pattern

```arkts
class Position {
  x: number = 0;
  y: number = 0;
}

@Component
struct MyComponent {
  @State position: Position = new Position();
  // title 未发生变化，改为普通变量
  title: string = "Welcome";

  build() {
    Column() {
      Text(this.title)
        .fontSize(24)
      Button("Move")
        .onClick(() => {
          this.position.x += 10;
          this.position.y += 10;
        })
    }
    .position({
      x: this.position.x,
      y: this.position.y
    })
  }
}
```

### Rationale

变量 title 在组件生命周期内没有被修改，仅用于显示，因此不需要使用状态变量 @State

## Example 2: `pair_482f4cf350b56582`

建议移除未改变的状态变量设置。

通用丢帧场景下，建议优先修改。

### Triggering pattern

```arkts
class User {
  id: number = 1;
  email: string = "user@example.com";
}

@Component
struct Dashboard {
  @State currentUser: User = new User();
  @State welcomeMessage: string = "Welcome to the Dashboard";

  build() {
    Column() {
      Text(this.welcomeMessage)
        .fontSize(22)
      // 显示用户信息
      UserInfo(this.currentUser)
    }
    .padding(10)
  }
}
```

### Repair pattern

```arkts
class User {
  id: number = 1;
  email: string = "user@example.com";
}

@Component
struct Dashboard {
  @State currentUser: User = new User();
  // welcomeMessage 未发生变化，改为普通变量
  welcomeMessage: string = "Welcome to the Dashboard";

  build() {
    Column() {
      Text(this.welcomeMessage)
        .fontSize(22)
      // 显示用户信息
      UserInfo(this.currentUser)
    }
    .padding(10)
  }
}
```

### Rationale

变量 welcomeMessage 在组件生命周期内没有被修改，仅用于显示欢迎信息，因此不需要使用状态变量 @State。将其更改为普通变量可以减少不必要的状态管理，提升性能。

## Example 3: `pair_85f719613a5ddb49`

建议移除未改变的状态变量设置。

通用丢帧场景下，建议优先修改。

### Triggering pattern

```arkts
@Observed
class Translate {
  translateX: number = 20;
}
@Component
struct Title {
  build() {
    Row() {
      // 本地资源 icon.png
      Image($r('app.media.icon'))
        .width(50)
        .height(50)
      Text("Title")
        .fontSize(20)
    }
  }
}
@Entry
@Component
struct MyComponent{
  @State translateObj: Translate = new Translate();
  @State button_msg: string = "i am button";

  build() {
    Column() {
      Title()
      Stack() {
      }
      .backgroundColor("black")
      .width(200)
      .height(400)
      // 这里只是用了状态变量button_msg的值，没有任何写的操作
      Button(this.button_msg)
        .onClick(() => {
          animateTo({
            duration: 50
          },()=>{
            this.translateObj.translateX = (this.translateObj.translateX + 50) % 150
          })
        })
    }
    .translate({
      x: this.translateObj.translateX
    })
  }
}
```

### Repair pattern

```arkts
class Translate {
  translateX: number = 20;
}

@Component
struct Title {
  build() {
    Row() {
      // 本地资源 icon.png
      Image($r('app.media.icon')) 
        .width(50)
        .height(50)
      Text("Title")
        .fontSize(20)
    }
  }
}

@Entry
@Component
struct MyComponent{
  @State translateObj: Translate = new Translate();
  // 直接使用一般变量即可
  button_msg: string = "i am button";

  build() {
    Column() {
      Title()
      Stack() {
      }
      .backgroundColor("black")
      .width(200)
      .height(400)

      Button(this.button_msg)
        .onClick(() => {
          animateTo({
            duration: 50
          }, () => {
            this.translateObj.translateX = (this.translateObj.translateX + 50) % 150
          })
        })
    }
    .translate({
      x: this.translateObj.translateX
    })
  }
}
```

### Rationale

button_msg没有改变，只有使用，因此不需要使用状态变量，使用普通变量即可

## Example 4: `pair_92baf2405911c161`

建议移除未改变的状态变量设置。

通用丢帧场景下，建议优先修改。

### Triggering pattern

```arkts
class Settings {
  theme: string = "dark";
  notificationsEnabled: boolean = true;
}

@Component
struct UserProfile {
  @State userSettings: Settings = new Settings();
  @State userName: string = "John Doe";

  build() {
    Column() {
      Text("User: " + this.userName)
        .fontSize(18)
      Toggle("Enable Notifications", this.userSettings.notificationsEnabled)
        .onChange((value) => {
          this.userSettings.notificationsEnabled = value;
        })
    }
    .padding(16)
  }
}
```

### Repair pattern

```arkts
class Settings {
  theme: string = "dark";
  notificationsEnabled: boolean = true;
}

@Component
struct UserProfile {
  @State userSettings: Settings = new Settings();
  // userName 未发生变化，改为普通变量
  userName: string = "John Doe";

  build() {
    Column() {
      Text("User: " + this.userName)
        .fontSize(18)
      Toggle("Enable Notifications", this.userSettings.notificationsEnabled)
        .onChange((value) => {
          this.userSettings.notificationsEnabled = value;
        })
    }
    .padding(16)
  }
}
```

### Rationale

变量 userName 在组件生命周期内没有被修改，仅用于显示用户名称，因此不需要使用状态变量 @State。将其更改为普通变量可以减少不必要的状态管理，提升性能。

## Example 5: `pair_da789332f5165b3f`

建议移除未改变的状态变量设置。

通用丢帧场景下，建议优先修改。

### Triggering pattern

```arkts
class Config {
  apiEndpoint: string = "https://api.example.com";
  timeout: number = 5000;
}

@Entry
@Component
struct AppConfig {
  @State config: Config = new Config();
  @State appTitle: string = "My Application";

  build() {
    Column() {
      Header(this.appTitle)
      // 配置详情显示
      ConfigView(this.config)
    }
    .padding(20)
  }
}
```

### Repair pattern

```arkts
class Config {
  apiEndpoint: string = "https://api.example.com";
  timeout: number = 5000;
}

@Entry
@Component
struct AppConfig {
  @State config: Config = new Config();
  // appTitle 未发生变化，改为普通变量
  appTitle: string = "My Application";

  build() {
    Column() {
      Header(this.appTitle)
      // 配置详情显示
      ConfigView(this.config)
    }
    .padding(20)
  }
}
```

### Rationale

变量 appTitle 在组件生命周期内没有被修改，仅用于显示应用标题，因此不需要使用状态变量 @State。将其更改为普通变量可以减少不必要的状态管理，提升性能。
