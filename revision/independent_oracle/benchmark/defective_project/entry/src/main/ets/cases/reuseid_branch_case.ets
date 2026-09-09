import { ChatInfoEntry, ChatItem, PublicChatItem } from './reuseid_branch_components';

class ChatDataSource implements IDataSource {
  private rows: ChatInfoEntry[] = [];
  private listeners: DataChangeListener[] = [];

  totalCount(): number { return this.rows.length; }
  getData(index: number): ChatInfoEntry { return this.rows[index]; }
  getReuseIdByIndex(index: number): string {
    return this.rows[index].isPublicChat ? 'public-chat' : 'private-chat';
  }
  registerDataChangeListener(listener: DataChangeListener): void { this.listeners.push(listener); }
  unregisterDataChangeListener(listener: DataChangeListener): void {
    const position: number = this.listeners.indexOf(listener);
    if (position >= 0) { this.listeners.splice(position, 1); }
  }
}

@Reusable
@Component
struct ChatRowView {
  @State chatInfo: ChatInfoEntry = new ChatInfoEntry();

  aboutToReuse(params: Record<string, ESObject>): void {
    this.chatInfo = params.chatInfo as ChatInfoEntry;
  }

  build() {
    Button() {
      Row() {
        if (this.chatInfo['isPublicChat']) {
          PublicChatItem({ chatInfo: this.chatInfo })
        } else {
          ChatItem({ chatInfo: this.chatInfo })
        }
      }
    }
  }
}

@Entry
@Component
struct ChatTimeline {
  private rows: ChatDataSource = new ChatDataSource();

  build() {
    List() {
      LazyForEach(this.rows, (item: ChatInfoEntry) => {
        ListItem() {
          ChatRowView({ chatInfo: item })
        }
      }, (item: ChatInfoEntry) => item.id)
    }
    .cachedCount(3)
    .width('100%')
    .height(320)
  }
}
