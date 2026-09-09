export class ChatInfoEntry {
  id: string = '';
  isPublicChat: boolean = false;
}

@Reusable
@Component
export struct PublicChatItem {
  @State chatInfo: ChatInfoEntry = new ChatInfoEntry();

  aboutToReuse(params: Record<string, ESObject>): void {
    this.chatInfo = params.chatInfo as ChatInfoEntry;
  }

  build() { Text(`Public ${this.chatInfo.id}`) }
}

@Reusable
@Component
export struct ChatItem {
  @State chatInfo: ChatInfoEntry = new ChatInfoEntry();

  aboutToReuse(params: Record<string, ESObject>): void {
    this.chatInfo = params.chatInfo as ChatInfoEntry;
  }

  build() { Text(`Private ${this.chatInfo.id}`) }
}
