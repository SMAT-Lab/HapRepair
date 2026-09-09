# @performance/multiple-associations-state-var-check

Static repair references: 15. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1163f2f62ae50f4a`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class DisplaySettings {
  darkMode: boolean = false;
}

@Entry
@Component
struct DemoG {
  @State display: DisplaySettings = new DisplaySettings();

  build() {
    Column() {
      Navbar({ display: this.display })
      Sidebar({ display: this.display })
      Text('Toggle Dark Mode')
        .onClick(() => {
          this.display.darkMode = !this.display.darkMode;
        })
    }
  }
}

@Component
struct Navbar {
  @Link display: DisplaySettings;

  build() {
    Text('Navbar')
      .backgroundColor(this.display.darkMode ? '#333333' : '#eeeeee')
  }
}

@Component
struct Sidebar {
  @Link display: DisplaySettings;

  build() {
    Text('Sidebar')
      .backgroundColor(this.display.darkMode ? '#333333' : '#eeeeee')
  }
}
```

### Repair pattern

```arkts
@Observed
class DisplaySettings {
  darkMode: boolean = false;
}

@Entry
@Component
struct DemoGFixed {
  @State display: DisplaySettings = new DisplaySettings();

  build() {
    Column() {
      Navbar({ display: this.display })
      Sidebar({ display: this.display })
      Text('Toggle Dark Mode')
        .onClick(() => {
          this.display.darkMode = !this.display.darkMode;
        })
    }
  }
}

@Component
struct Navbar {
  @Link @Watch('onModeChange') display: DisplaySettings;
  @State localDarkMode: boolean = false;

  onModeChange() {
    this.localDarkMode = this.display.darkMode;
  }

  build() {
    Text('Navbar')
      .backgroundColor(this.localDarkMode ? '#333333' : '#eeeeee')
  }
}

@Component
struct Sidebar {
  @Link @Watch('onModeChange') display: DisplaySettings;
  @State localDarkMode: boolean = false;

  onModeChange() {
    this.localDarkMode = this.display.darkMode;
  }

  build() {
    Text('Sidebar')
      .backgroundColor(this.localDarkMode ? '#333333' : '#eeeeee')
  }
}
```

### Rationale

darkMode 属性变更会导致 Navbar 和 Sidebar 不必要的重新渲染。

## Example 2: `pair_21b6f42c0c9f6c79`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example2 {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompC({ subStyle: this.uiStyle })
      Text('Toggle Checked')
        .onClick(() => {
          this.uiStyle.isChecked = !this.uiStyle.isChecked;
        })
    }
  }
}

@Component
struct CompC {
  @Link subStyle: UIStyle;

  build() {
    if (this.subStyle.isChecked) {
      Text('Checked')
    } else {
      Text('Unchecked')
    }
  }
}
```

### Repair pattern

```arkts
@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example2Fixed {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompC({ subStyle: this.uiStyle })
      Text('Toggle Checked')
        .onClick(() => {
          this.uiStyle.isChecked = !this.uiStyle.isChecked;
        })
    }
  }
}

@Component
struct CompC {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localChecked: boolean = false;

  onStyleChange() {
    this.localChecked = this.subStyle.isChecked;
  }

  build() {
    if (this.localChecked) {
      Text('Checked')
    } else {
      Text('Unchecked')
    }
  }
}
```

### Rationale

isChecked 的变化会导致全部组件 CompC 重新渲染，即使组件的可见性状态未改变。这可以通过监听机制优化。

## Example 3: `pair_3893876d767556b1`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class UIStyle {
  fontColor: string = '#000000';
}

@Entry
@Component
struct Example3 {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompA({ subStyle: this.uiStyle })
      CompC({ subStyle: this.uiStyle })
      Text('Change Font Color')
        .onClick(() => {
          this.uiStyle.fontColor = '#FF0000';
        })
    }
  }
}

@Component
struct CompA {
  @Link subStyle: UIStyle;

  build() {
    Text('Component A')
      .fontColor(this.subStyle.fontColor)
  }
}

@Component
struct CompC {
  @Link subStyle: UIStyle;

  build() {
    Text('Component C')
      .fontColor(this.subStyle.fontColor)
  }
}
```

### Repair pattern

```arkts
@Observed
class UIStyle {
  fontColor: string = '#000000';
}

@Entry
@Component
struct Example3Fixed {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompA({ subStyle: this.uiStyle })
      CompC({ subStyle: this.uiStyle })
      Text('Change Font Color')
        .onClick(() => {
          this.uiStyle.fontColor = '#FF0000';
        })
    }
  }
}

@Component
struct CompA {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontColor: string = '#000000';

  onStyleChange() {
    this.localFontColor = this.subStyle.fontColor;
  }

  build() {
    Text('Component A')
      .fontColor(this.localFontColor)
  }
}

@Component
struct CompC {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontColor: string = '#000000';

  onStyleChange() {
    this.localFontColor = this.subStyle.fontColor;
  }

  build() {
    Text('Component C')
      .fontColor(this.localFontColor)
  }
}
```

### Rationale

当 fontColor 属性改变时，所有使用该属性的组件都会更新，从而导致不必要的组件渲染。

## Example 4: `pair_3b588e290edc4c35`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class ViewMode {
  displayMode: string = 'card';
}

@Entry
@Component
struct DemoD {
  @State view: ViewMode = new ViewMode();

  build() {
    Column() {
      ListView({ mode: this.view })
      GridView({ mode: this.view })
      Text('Switch Display Mode')
        .onClick(() => {
          this.view.displayMode = 'list';
        })
    }
  }
}

@Component
struct ListView {
  @Link mode: ViewMode;

  build() {
    Text('ListView')
      .visible(this.mode.displayMode === 'list')
  }
}

@Component
struct GridView {
  @Link mode: ViewMode;

  build() {
    Text('GridView')
      .visible(this.mode.displayMode === 'card')
  }
}
```

### Repair pattern

```arkts
@Observed
class ViewMode {
  displayMode: string = 'card';
}

@Entry
@Component
struct DemoDFixed {
  @State view: ViewMode = new ViewMode();

  build() {
    Column() {
      ListView({ mode: this.view })
      GridView({ mode: this.view })
      Text('Switch Display Mode')
        .onClick(() => {
          this.view.displayMode = 'list';
        })
    }
  }
}

@Component
struct ListView {
  @Link @Watch('onModeChange') mode: ViewMode;
  @State localDisplayMode: string = 'card';

  onModeChange() {
    this.localDisplayMode = this.mode.displayMode;
  }

  build() {
    Text('ListView')
      .visible(this.localDisplayMode === 'list')
  }
}

@Component
struct GridView {
  @Link @Watch('onModeChange') mode: ViewMode;
  @State localDisplayMode: string = 'card';

  onModeChange() {
    this.localDisplayMode = this.mode.displayMode;
  }

  build() {
    Text('GridView')
      .visible(this.localDisplayMode === 'card')
  }
}
```

### Rationale

displayMode的修改会导致ListView和GridView组件不必要的重新渲染。

## Example 5: `pair_461c2968ebb4d357`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class NetworkConfig {
  isConnected: boolean = true;
}

@Entry
@Component
struct DemoJ {
  @State network: NetworkConfig = new NetworkConfig();

  build() {
    Row() {
      StatusIndicator({ network: this.network })
      DataUsage({ network: this.network })
      Text('Connect/Disconnect')
        .onClick(() => {
          this.network.isConnected = !this.network.isConnected;
        })
    }
  }
}

@Component
struct StatusIndicator {
  @Link network: NetworkConfig;

  build() {
    Text('Status Indicator')
      .text(this.network.isConnected ? 'Connected' : 'Disconnected')
  }
}

@Component
struct DataUsage {
  @Link network: NetworkConfig;

  build() {
    Text('Data Usage')
      .text(this.network.isConnected ? 'Tracking Usage' : 'Offline Mode')
  }
}
```

### Repair pattern

```arkts
@Observed
class NetworkConfig {
  isConnected: boolean = true;
}

@Entry
@Component
struct DemoJFixed {
  @State network: NetworkConfig = new NetworkConfig();

  build() {
    Row() {
      StatusIndicator({ network: this.network })
      DataUsage({ network: this.network })
      Text('Connect/Disconnect')
        .onClick(() => {
          this.network.isConnected = !this.network.isConnected;
        })
    }
  }
}

@Component
struct StatusIndicator {
  @Link @Watch('onConnectionChange') network: NetworkConfig;
  @State localIsConnected: boolean = true;

  onConnectionChange() {
    this.localIsConnected = this.network.isConnected;
  }

  build() {
    Text('Status Indicator')
      .text(this.localIsConnected ? 'Connected' : 'Disconnected')
  }
}

@Component
struct DataUsage {
  @Link @Watch('onConnectionChange') network: NetworkConfig;
  @State localIsConnected: boolean = true;

  onConnectionChange() {
    this.localIsConnected = this.network.isConnected;
  }

  build() {
    Text('Data Usage')
      .text(this.localIsConnected ? 'Tracking Usage' : 'Offline Mode')
  }
}
```

### Rationale

isConnected 属性变化引起 StatusIndicator 和 DataUsage 的不必要重新渲染。

## Example 6: `pair_5888563ec75d7a69`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class WindowStyle {
  isFullScreen: boolean = false;
}

@Entry
@Component
struct DemoB {
  @State style: WindowStyle = new WindowStyle();

  build() {
    Row() {
      SideBar({ style: this.style })
      MainContent({ style: this.style })
      Text('Toggle Full Screen')
        .onClick(() => {
          this.style.isFullScreen = !this.style.isFullScreen;
        })
    }
  }
}

@Component
struct SideBar {
  @Link style: WindowStyle;

  build() {
    if (this.style.isFullScreen) {
      Text('SideBar Hidden')
    } else {
      Text('SideBar Visible')
    }
  }
}

@Component
struct MainContent {
  @Link style: WindowStyle;

  build() {
    if (this.style.isFullScreen) {
      Text('FullScreen Mode')
    } else {
      Text('Normal Mode')
    }
  }
}
```

### Repair pattern

```arkts
@Observed
class WindowStyle {
  isFullScreen: boolean = false;
}

@Entry
@Component
struct DemoBFixed {
  @State style: WindowStyle = new WindowStyle();

  build() {
    Row() {
      SideBar({ style: this.style })
      MainContent({ style: this.style })
      Text('Toggle Full Screen')
        .onClick(() => {
          this.style.isFullScreen = !this.style.isFullScreen;
        })
    }
  }
}

@Component
struct SideBar {
  @Link @Watch('onStyleChange') style: WindowStyle;
  @State localFullScreen: boolean = false;

  onStyleChange() {
    this.localFullScreen = this.style.isFullScreen;
  }

  build() {
    if (this.localFullScreen) {
      Text('SideBar Hidden')
    } else {
      Text('SideBar Visible')
    }
  }
}

@Component
struct MainContent {
  @Link @Watch('onStyleChange') style: WindowStyle;
  @State localFullScreen: boolean = false;

  onStyleChange() {
    this.localFullScreen = this.style.isFullScreen;
  }

  build() {
    if (this.localFullScreen) {
      Text('FullScreen Mode')
    } else {
      Text('Normal Mode')
    }
  }
}
```

### Rationale

isFullScreen 属性的变化会导致SideBar和MainContent组件不必要的重新渲染。

## Example 7: `pair_62579ce8f1083132`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class UIStyle {
  fontSize: number = 0;
}

@Entry
@Component
struct Example1 {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompA({ subStyle: this.uiStyle })
      CompB({ subStyle: this.uiStyle })
      Text('Change Font Size')
        .onClick(() => {
          this.uiStyle.fontSize = 24;
        })
    }
  }
}

@Component
struct CompA {
  @Link subStyle: UIStyle;

  build() {
    Text('Component A')
      .fontSize(this.subStyle.fontSize)
  }
}

@Component
struct CompB {
  @Link subStyle: UIStyle;

  build() {
    Text('Component B')
      .fontSize(this.subStyle.fontSize)
  }
}
```

### Repair pattern

```arkts
@Observed
class UIStyle {
  fontSize: number = 0;
}

@Entry
@Component
struct Example1Fixed {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompA({ subStyle: this.uiStyle })
      CompB({ subStyle: this.uiStyle })
      Text('Change Font Size')
        .onClick(() => {
          this.uiStyle.fontSize = 24;
        })
    }
  }
}

@Component
struct CompA {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontSize: number = 0;

  onStyleChange() {
    this.localFontSize = this.subStyle.fontSize;
  }

  build() {
    Text('Component A')
      .fontSize(this.localFontSize)
  }
}

@Component
struct CompB {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontSize: number = 0;

  onStyleChange() {
    this.localFontSize = this.subStyle.fontSize;
  }

  build() {
    Text('Component B')
      .fontSize(this.localFontSize)
  }
}
```

### Rationale

fontSize 的修改导致所有使用了该数据属性的组件都重新渲染，即使它们的显示可能没有改变。这会导致不必要的性能开销。

## Example 8: `pair_6c65d0811b1c335d`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class UIConfig {
  layoutType: string = 'grid';
}

@Entry
@Component
struct DemoA {
  @State config: UIConfig = new UIConfig();

  build() {
    Column() {
      HeaderComp({ layout: this.config })
      FooterComp({ layout: this.config })
      Text('Switch Layout')
        .onClick(() => {
          this.config.layoutType = 'list';
        })
    }
  }
}

@Component
struct HeaderComp {
  @Link layout: UIConfig;

  build() {
    Text('Header')
      .textAlign(this.layout.layoutType === 'grid' ? 'center' : 'left')
  }
}

@Component
struct FooterComp {
  @Link layout: UIConfig;

  build() {
    Text('Footer')
      .textAlign(this.layout.layoutType === 'grid' ? 'center' : 'left')
  }
}
```

### Repair pattern

```arkts
@Observed
class UIConfig {
  layoutType: string = 'grid';
}

@Entry
@Component
struct DemoAFixed {
  @State config: UIConfig = new UIConfig();

  build() {
    Column() {
      HeaderComp({ layout: this.config })
      FooterComp({ layout: this.config })
      Text('Switch Layout')
        .onClick(() => {
          this.config.layoutType = 'list';
        })
    }
  }
}

@Component
struct HeaderComp {
  @Link @Watch('onLayoutChange') layout: UIConfig;
  @State localLayoutType: string = 'grid';

  onLayoutChange() {
    this.localLayoutType = this.layout.layoutType;
  }

  build() {
    Text('Header')
      .textAlign(this.localLayoutType === 'grid' ? 'center' : 'left')
  }
}

@Component
struct FooterComp {
  @Link @Watch('onLayoutChange') layout: UIConfig;
  @State localLayoutType: string = 'grid';

  onLayoutChange() {
    this.localLayoutType = this.layout.layoutType;
  }

  build() {
    Text('Footer')
      .textAlign(this.localLayoutType === 'grid' ? 'center' : 'left')
  }
}
```

### Rationale

layoutType 的修改会导致所有使用该数据属性的组件重新渲染，从而带来不必要的性能开销。

## Example 9: `pair_81e383b94a6c8111`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class NotificationSettings {
  enableSound: boolean = true;
}

@Entry
@Component
struct DemoE {
  @State settings: NotificationSettings = new NotificationSettings();

  build() {
    Column() {
      SoundControl({ settings: this.settings })
      VibrationControl({ settings: this.settings })
      Text('Toggle Sound')
        .onClick(() => {
          this.settings.enableSound = !this.settings.enableSound;
        })
    }
  }
}

@Component
struct SoundControl {
  @Link settings: NotificationSettings;

  build() {
    if (this.settings.enableSound) {
      Text('Sound On')
    } else {
      Text('Sound Off')
    }
  }
}

@Component
struct VibrationControl {
  @Link settings: NotificationSettings;

  build() {
    if (this.settings.enableSound) {
      Text('Vibration Enabled')
    } else {
      Text('Vibration Disabled')
    }
  }
}
```

### Repair pattern

```arkts
@Observed
class NotificationSettings {
  enableSound: boolean = true;
}

@Entry
@Component
struct DemoEFixed {
  @State settings: NotificationSettings = new NotificationSettings();

  build() {
    Column() {
      SoundControl({ settings: this.settings })
      VibrationControl({ settings: this.settings })
      Text('Toggle Sound')
        .onClick(() => {
          this.settings.enableSound = !this.settings.enableSound;
        })
    }
  }
}

@Component
struct SoundControl {
  @Link @Watch('onSettingsChange') settings: NotificationSettings;
  @State localEnableSound: boolean = true;

  onSettingsChange() {
    this.localEnableSound = this.settings.enableSound;
  }

  build() {
    if (this.localEnableSound) {
      Text('Sound On')
    } else {
      Text('Sound Off')
    }
  }
}

@Component
struct VibrationControl {
  @Link @Watch('onSettingsChange') settings: NotificationSettings;
  @State localEnableSound: boolean = true;

  onSettingsChange() {
    this.localEnableSound = this.settings.enableSound;
  }

  build() {
    if (this.localEnableSound) {
      Text('Vibration Enabled')
    } else {
      Text('Vibration Disabled')
    }
  }
}
```

### Rationale

enableSound 切换时，会导致所有相关组件不必要地重新渲染。

## Example 10: `pair_8cfea566bfac4650`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class ProfileSettings {
  notificationsEnabled: boolean = true;
}

@Entry
@Component
struct DemoF {
  @State profile: ProfileSettings = new ProfileSettings();

  build() {
    Row() {
      NotificationComp({ profile: this.profile })
      AlertComp({ profile: this.profile })
      Text('Toggle Notifications')
        .onClick(() => {
          this.profile.notificationsEnabled = !this.profile.notificationsEnabled;
        })
    }
  }
}

@Component
struct NotificationComp {
  @Link profile: ProfileSettings;

  build() {
    if (this.profile.notificationsEnabled) {
      Text('Notifications On')
    } else {
      Text('Notifications Off')
    }
  }
}

@Component
struct AlertComp {
  @Link profile: ProfileSettings;

  build() {
    if (this.profile.notificationsEnabled) {
      Text('Alerts Enabled')
    } else {
      Text('Alerts Disabled')
    }
  }
}
```

### Repair pattern

```arkts
@Observed
class ProfileSettings {
  notificationsEnabled: boolean = true;
}

@Entry
@Component
struct DemoF {
  @State profile: ProfileSettings = new ProfileSettings();

  build() {
    Row() {
      NotificationComp({ profile: this.profile })
      AlertComp({ profile: this.profile })
      Text('Toggle Notifications')
        .onClick(() => {
          this.profile.notificationsEnabled = !this.profile.notificationsEnabled;
        })
    }
  }
}

@Component
struct NotificationComp {
  @Link @Watch('onNotifyChange') profile: ProfileSettings;
  @State localNotificationsEnabled: boolean = true;

  onNotifyChange() {
    this.localNotificationsEnabled = this.profile.notificationsEnabled;
  }

  build() {
    if (this.localNotificationsEnabled) {
      Text('Notifications On')
    } else {
      Text('Notifications Off')
    }
  }
}

@Component
struct AlertComp {
  @Link @Watch('onNotifyChange') profile: ProfileSettings;
  @State localNotificationsEnabled: boolean = true;

  onNotifyChange() {
    this.localNotificationsEnabled = this.profile.notificationsEnabled;
  }

  build() {
    if (this.localNotificationsEnabled) {
      Text('Alerts Enabled')
    } else {
      Text('Alerts Disabled')
    }
  }
}
```

### Rationale

notificationsEnabled 属性的变化会导致 NotificationComp 和 AlertComp 组件不必要的重新渲染。

## Example 11: `pair_946c10f9140bdf6d`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class ThemeSettings {
  themeColor: string = 'light';
}

@Entry
@Component
struct DemoC {
  @State theme: ThemeSettings = new ThemeSettings();

  build() {
    Column() {
      Header({ theme: this.theme })
      Footer({ theme: this.theme })
      Text('Switch Theme')
        .onClick(() => {
          this.theme.themeColor = 'dark';
        })
    }
  }
}

@Component
struct Header {
  @Link theme: ThemeSettings;

  build() {
    Text('Header')
      .backgroundColor(this.theme.themeColor === 'light' ? '#ffffff' : '#000000')
  }
}

@Component
struct Footer {
  @Link theme: ThemeSettings;

  build() {
    Text('Footer')
      .backgroundColor(this.theme.themeColor === 'light' ? '#ffffff' : '#000000')
  }
}
```

### Repair pattern

```arkts
@Observed
class ThemeSettings {
  themeColor: string = 'light';
}

@Entry
@Component
struct DemoCFixed {
  @State theme: ThemeSettings = new ThemeSettings();

  build() {
    Column() {
      Header({ theme: this.theme })
      Footer({ theme: this.theme })
      Text('Switch Theme')
        .onClick(() => {
          this.theme.themeColor = 'dark';
        })
    }
  }
}

@Component
struct Header {
  @Link @Watch('onThemeChange') theme: ThemeSettings;
  @State localThemeColor: string = 'light';

  onThemeChange() {
    this.localThemeColor = this.theme.themeColor;
  }

  build() {
    Text('Header')
      .backgroundColor(this.localThemeColor === 'light' ? '#ffffff' : '#000000')
  }
}

@Component
struct Footer {
  @Link @Watch('onThemeChange') theme: ThemeSettings;
  @State localThemeColor: string = 'light';

  onThemeChange() {
    this.localThemeColor = this.theme.themeColor;
  }

  build() {
    Text('Footer')
      .backgroundColor(this.localThemeColor === 'light' ? '#ffffff' : '#000000')
  }
}
```

### Rationale

themeColor 变更时，会导致不必要的Header和Footer重新渲染。

## Example 12: `pair_a68debca2ba2e41e`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class UIStyle {
  fontSize: number = 0;
  fontColor: string = '';
  isChecked: boolean = false;
}
@Entry
@Component
struct MultipleAssociationsStateVarReport0 {
  @State uiStyle: UIStyle = new UIStyle();
  private listData: string[] = [];
  aboutToAppear(): void {
    for (let i = 0; i < 10; i++) {
      this.listData.push(`ListItemComponent ${i}`);
    }
  }
  build() {
    Row() {
      Column() {
        CompA({item: '1', index: 1, subStyle: this.uiStyle})
        CompB({item: '2', index: 2, subStyle: this.uiStyle})
        CompC({item: '3', index: 3, subStyle: this.uiStyle})
        Text('change state var')
          .onClick(()=>{
            this.uiStyle.fontSize = 20;
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}
@Component
struct CompA {
  @Prop item: string;
  @Prop index: number;
  @Link subStyle: UIStyle;
  private sizeFont: number = 50;
  isRender(): number {
    console.info(`CompA ${this.index} Text is rendered`);
    return this.sizeFont;
  }
  build() {
    Column() {
      Text(this.item)
        .fontSize(this.isRender())
        .fontSize(this.subStyle.fontSize)
      Text('abc')
    }
  }
}
@Component
struct CompB {
  @Prop item: string;
  @Prop index: number;
  @Link subStyle: UIStyle;
  private sizeFont: number = 50;
  isRender(): number {
    console.info(`CompB ${this.index} Text is rendered`);
    return this.sizeFont;
  }
  build() {
    Column() {
      Text(this.item)
        .fontSize(this.isRender())
        .fontColor(this.subStyle.fontColor)
      Text('abc')
    }
  }
}
@Component
struct CompC {
  @Prop item: string;
  @Prop index: number;
  @Link subStyle: UIStyle;
  private sizeFont: number = 50;
  isRender(): number {
    console.info(`CompC ${this.index} Text is rendered`);
    return this.sizeFont;
  }
  build() {
    Column() {
      if (this.subStyle.isChecked) {
        Text('checked')
      } else {
        Text('unchecked')
      }
    }
  }
}
```

### Repair pattern

```arkts
@Observed
class UIStyle {
  fontSize: number = 0;
  fontColor: string = '';
  isChecked: boolean = false;
}
@Entry
@Component
struct MultipleAssociationsStateVarNoReport0 {
  @State uiStyle: UIStyle = new UIStyle();
  private listData: string[] = [];
  aboutToAppear(): void {
    for (let i = 0; i < 10; i++) {
      this.listData.push(`ListItemComponent ${i}`);
    }
  }
  build() {
    Row() {
      Column() {
        CompA({item: '1', index: 1, subStyle: this.uiStyle})
        CompB({item: '2', index: 2, subStyle: this.uiStyle})
        CompC({item: '3', index: 3, subStyle: this.uiStyle})
        Text('change state var')
          .onClick(()=>{
            this.uiStyle.fontSize = 20;
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}
@Component
struct CompA {
  @Prop item: string;
  @Prop index: number;
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State fontSize: number = 0;
  isRender(): number {
    console.info(`CompA ${this.index} Text is rendered`);
    return this.fontSize;
  }
  onStyleChange() {
    this.fontSize = this.subStyle.fontSize;
  }
  build() {
    Column() {
      Text(this.item)
        .fontSize(this.isRender())
        .fontSize(this.fontSize)
      Text('abc')
    }
  }
}
@Component
struct CompB {
  @Prop item: string;
  @Prop index: number;
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State fontColor: string = '#00ffff';
  isRender(): number {
    console.info(`CompB ${this.index} Text is rendered`);
    return 10;
  }
  onStyleChange() {
    this.fontColor = this.subStyle.fontColor;
  }
  build() {
    Column() {
      Text(this.item)
        .fontSize(this.isRender())
        .fontColor(this.fontColor)
      Text('abc')
    }
  }
}
@Component
struct CompC {
  @Prop item: string;
  @Prop index: number;
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State isChecked: boolean = false;
  isRender(): number {
    console.info(`CompC ${this.index} Text is rendered`);
    return 50;
  }
  onStyleChange() {
    this.isChecked = this.subStyle.isChecked;
  }
  build() {
    Column() {
      if (this.isChecked) {
        Text('checked')
      } else {
        Text('unchecked')
      }
    }
  }
}
```

### Rationale

CompA, CompB, CompC同时关联this.uiStyle, 当该属性发生变化时，@Watch可以监听变化并执行相应的函数，避免不必要的组件更新

## Example 13: `pair_a9200fe71c2d28fb`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class AudioState {
  mute: boolean = false;
}

@Entry
@Component
struct DemoH {
  @State audio: AudioState = new AudioState();

  build() {
    Row() {
      VolumeControl({ audio: this.audio })
      MuteIndicator({ audio: this.audio })
      Text('Mute/Unmute')
        .onClick(() => {
          this.audio.mute = !this.audio.mute;
        })
    }
  }
}

@Component
struct VolumeControl {
  @Link audio: AudioState;

  build() {
    Text('Volume Control')
      .opacity(this.audio.mute ? 0.5 : 1.0)
  }
}

@Component
struct MuteIndicator {
  @Link audio: AudioState;

  build() {
    Text('Mute Status')
      .opacity(this.audio.mute ? 1.0 : 0.5)
  }
}
```

### Repair pattern

```arkts
@Observed
class AudioState {
  mute: boolean = false;
}

@Entry
@Component
struct DemoHFixed {
  @State audio: AudioState = new AudioState();

  build() {
    Row() {
      VolumeControl({ audio: this.audio })
      MuteIndicator({ audio: this.audio })
      Text('Mute/Unmute')
        .onClick(() => {
          this.audio.mute = !this.audio.mute;
        })
    }
  }
}

@Component
struct VolumeControl {
  @Link @Watch('onMuteChange') audio: AudioState;
  @State localMute: boolean = false;

  onMuteChange() {
    this.localMute = this.audio.mute;
  }

  build() {
    Text('Volume Control')
      .opacity(this.localMute ? 0.5 : 1.0)
  }
}

@Component
struct MuteIndicator {
  @Link @Watch('onMuteChange') audio: AudioState;
  @State localMute: boolean = false;

  onMuteChange() {
    this.localMute = this.audio.mute;
  }

  build() {
    Text('Mute Status')
      .opacity(this.localMute ? 1.0 : 0.5)
  }
}
```

### Rationale

mute 属性变化引起 VolumeControl 和 MuteIndicator 的不必要重新渲染。

## Example 14: `pair_dbddfb12c40a0e15`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class BatteryStatus {
  isCharging: boolean = true;
}

@Entry
@Component
struct DemoI {
  @State battery: BatteryStatus = new BatteryStatus();

  build() {
    Column() {
      BatteryIcon({ battery: this.battery })
      BatteryInfo({ battery: this.battery })
      Text('Toggle Charging')
        .onClick(() => {
          this.battery.isCharging = !this.battery.isCharging;
        })
    }
  }
}

@Component
struct BatteryIcon {
  @Link battery: BatteryStatus;

  build() {
    Text('Battery Icon')
      .text(this.battery.isCharging ? 'Charging' : 'Discharging')
  }
}

@Component
struct BatteryInfo {
  @Link battery: BatteryStatus;

  build() {
    Text('Battery Info')
      .text(this.battery.isCharging ? 'Power Source Connected' : 'Running on Battery')
  }
}
```

### Repair pattern

```arkts
@Observed
class BatteryStatus {
  isCharging: boolean = true;
}

@Entry
@Component
struct DemoIFixed {
  @State battery: BatteryStatus = new BatteryStatus();

  build() {
    Column() {
      BatteryIcon({ battery: this.battery })
      BatteryInfo({ battery: this.battery })
      Text('Toggle Charging')
        .onClick(() => {
          this.battery.isCharging = !this.battery.isCharging;
        })
    }
  }
}

@Component
struct BatteryIcon {
  @Link @Watch('onChargeChange') battery: BatteryStatus;
  @State localIsCharging: boolean = true;

  onChargeChange() {
    this.localIsCharging = this.battery.isCharging;
  }

  build() {
    Text('Battery Icon')
      .text(this.localIsCharging ? 'Charging' : 'Discharging')
  }
}

@Component
struct BatteryInfo {
  @Link @Watch('onChargeChange') battery: BatteryStatus;
  @State localIsCharging: boolean = true;

  onChargeChange() {
    this.localIsCharging = this.battery.isCharging;
  }

  build() {
    Text('Battery Info')
      .text(this.localIsCharging ? 'Power Source Connected' : 'Running on Battery')
  }
}
```

### Rationale

isCharging 属性改变导致 BatteryIcon 和 BatteryInfo 不必要的重新渲染。

## Example 15: `pair_dc7bb4b3e2d20382`

多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。

### Triggering pattern

```arkts
@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example4 {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompB({ subStyle: this.uiStyle })
      CompC({ subStyle: this.uiStyle })
      Text('Toggle Check')
        .onClick(() => {
          this.uiStyle.isChecked = !this.uiStyle.isChecked;
        })
    }
  }
}

@Component
struct CompB {
  @Link subStyle: UIStyle;

  build() {
    if (this.subStyle.isChecked) {
      Text('Component B Checked')
    } else {
      Text('Component B Unchecked')
    }
  }
}

@Component
struct CompC {
  @Link subStyle: UIStyle;

  build() {
    if (this.subStyle.isChecked) {
      Text('Component C Checked')
    } else {
      Text('Component C Unchecked')
    }
  }
}
```

### Repair pattern

```arkts
@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example4Fixed {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompB({ subStyle: this.uiStyle })
      CompC({ subStyle: this.uiStyle })
      Text('Toggle Check')
        .onClick(() => {
          this.uiStyle.isChecked = !this.uiStyle.isChecked;
        })
    }
  }
}

@Component
struct CompB {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localChecked: boolean = false;

  onStyleChange() {
    this.localChecked = this.subStyle.isChecked;
  }

  build() {
    if (this.localChecked) {
      Text('Component B Checked')
    } else {
      Text('Component B Unchecked')
    }
  }
}

@Component
struct CompC {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localChecked: boolean = false;

  onStyleChange() {
    this.localChecked = this.subStyle.isChecked;
  }

  build() {
    if (this.localChecked) {
      Text('Component C Checked')
    } else {
      Text('Component C Unchecked')
    }
  }
}
```

### Rationale

当 isChecked 被切换时，无论其可见性状态如何，CompB 和 CompC 都会重新渲染，导致不必要的性能开销。
