// 建议使用reuseId标记不同结构的组件构成。
//
// 滑动丢帧场景下，建议优先修改。
import { MyDataSource } from './m/MyDataSource';
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
            // ListItem里有if-else并且直接在分支里使用了自定义复用组件
            Button({ type: ButtonType.Normal }) {
              Row() {
                if (item['isPublicChat']) {
                  // 源码文件，请以工程实际为准
                  PublicChatItem({ chatInfo: item as PublicChatItem })
                } else {
                  // 源码文件，请以工程实际为准
                  ChatItem({ chatInfo: item as ChatItem })
                }
              }.padding({ left: 16, right: 16 })
            }
            .type(ButtonType.Normal)
            .width('100%')
            .height('100%')
            .borderRadius(0)
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