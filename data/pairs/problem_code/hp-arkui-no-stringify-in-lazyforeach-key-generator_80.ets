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