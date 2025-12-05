
let storages = new LocalStorage();
storages.setOrCreate('eee', 22)

@Entry({storage: storages, useSharedStorage: false})
@Component
struct EntryOptionPage {
  @LocalStorageProp('eee') eee: number = 222;

  build() {
    Navigation() {
        Text(this.eee + '').id('textId')
    }
    .title('MainPage')
    .titleMode(NavigationTitleMode.Mini)
    .mode(NavigationMode.Stack)
  }
}