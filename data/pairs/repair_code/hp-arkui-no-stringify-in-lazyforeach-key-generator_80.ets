@Entry
@Component
struct App {
  build() {
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
    }, (listItem: userFileManager.FileAsset) => listItem.displayName.name)
  }
}