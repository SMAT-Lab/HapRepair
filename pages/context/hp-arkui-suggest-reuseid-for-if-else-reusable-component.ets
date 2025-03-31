
                if (item['isPublicChat']) {
                  // 源码文件，请以工程实际为准
                  PublicChatItem({ chatInfo: item as PublicChatItem })
                } else {
                  // 源码文件，请以工程实际为准
                  ChatItem({ chatInfo: item as ChatItem })
                }