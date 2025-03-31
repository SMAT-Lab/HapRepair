import hilog from '@ohos.hilog';
@Entry
@Component
struct UserInterface {
  build() {
    List()
      .onItemMove(() => {
        hilog.error(7003, 'UserInterface', 'onItemMove')
      })
  }
}