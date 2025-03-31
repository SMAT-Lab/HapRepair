// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';
import { ChartInfoEntry } from './data/DataEntry';
import { PublicChatItem } from './component/PublicChatItem';
import { ChatItem } from './component/ChatItem';

@Entry
@Component
struct MyComponent{
  private scroller: Scroller = new Scroller()
  private lazyChatList: MyDataSource = new MyDataSource();

  build() {
    Column() {
      List({ scroller: this.scroller }) {
        LazyForEach(this.lazyChatList, (item: ChartInfoEntry, index: number) => {
          ListItem() {
            // 使用reuseId进行组件复用的控制
            InnerRecentChat({ chatInfo: item }).reuseId(this.lazyChatList.getReuseIdByIndex(index))
          }
          .height(72)
        }, (item: ChartInfoEntry) => item.id)
      }
      .cachedCount(3)
      .width('100%')
      .height('100%')
    }
  }
}

@Reusable
@Component
struct InnerRecentChat {
  @State chatInfo: ChartInfoEntry = new ChartInfoEntry()

  aboutToReuse(params: Record<string, ESObject>): void {
    this.chatInfo = params.chatInfo as ChartInfoEntry
  }

  build() {
    Button({ type: ButtonType.Normal }) {
      Row() {
        if (this.chatInfo['isPublicChat']) {
          // 源码文件，请以工程实际为准
          PublicChatItem({ chatInfo: chatInfo as ChartInfoEntry })
        } else {
          // 源码文件，请以工程实际为准
          ChatItem({ chatInfo: this.chatInfo as ChatItem })
        }
      }.padding({ left: 16, right: 16 })
    }
    .type(ButtonType.Normal)
    .width('100%')
    .height('100%')
    .borderRadius(0)
  }
}