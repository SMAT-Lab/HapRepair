@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.fileList, (listItem: fileAccess.FileInfo) => {
        ListItem() {
          FileListItemComponent({
            fileListItem: listItem,
            itemFileList: $fileList,
            itemShowDeleteButton: $showDeleteButton,
            itemIsNoFile: $isNoFile
          })
        }
      }, (listItem: fileAccess.FileInfo) => listItem.fileName)
    }
  }
}

@Reuse
@Component
struct FileListItemComponent {
  @Link itemFileList: DocumentDataSource;
  @Link itemShowDeleteButton: boolean;
  @Link itemIsNoFile: boolean;
  private fileListItem: fileAccess.FileInfo = {} as fileAccess.FileInfo;
  private itemClickFunction: (fileAsset: fileAccess.FileInfo) => void = () => {
    prompt.showToast({
      duration: ONE_SECOND,
      message: $r('app.string.not_supported_tip')
    })
  }

  @Styles
  itemPressedStyles() {
    .backgroundColor($r('app.color.item_pressed'))
    .borderRadius(10)
  }

  build() {
      Row() {
        Image($r('app.media.default_document'))
          .objectFit(ImageFit.Fill)
          .width(40)
          .height(40)
          .margin({ left: 20 })

        Column() {
          Text(this.fileListItem.fileName)
            .maxLines(1)
            .width('75%')
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .fontSize(16)
            .fontColor($r('app.color.black'))

          Text(this.fileListItem.mtime + ' - ' + this.fileListItem.size + 'B')
            .fontSize(12)
            .margin({ top: 5 })
            .fontColor($r('app.color.font_gray'))
        }
        .margin({ left: 10 })
        .alignItems(HorizontalAlign.Start)
      }
  }
}
