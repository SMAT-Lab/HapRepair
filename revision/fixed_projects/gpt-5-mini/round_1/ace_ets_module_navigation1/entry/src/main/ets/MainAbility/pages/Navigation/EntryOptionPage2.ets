
let storages1 = new LocalStorage();
storages1.setOrCreate('eee', 22)

@Entry({storage: storages1, useSharedStorage: true})
@Component
struct EntryOptionPage2 {
  @LocalStorageProp('eee') eee: number = 222;

  build() {
    Navigation() {
      Text(this.eee + '').id('text2Id')
    }
    .title('MainPage')
    .titleMode(NavigationTitleMode.Mini)
    .mode(NavigationMode.Stack)
  }
}