
/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. All rights reserved.
 */

import { importSongs } from '../utils/importSongs/ImportSongs';
import { SongInfo } from '../utils/importSongs/InterfaceInfo';
import audioNapi from 'libentry.so';
import fileUri from '@ohos.file.fileuri';
import fs, { ReadOptions } from '@ohos.file.fs';
import { Logger } from '../utils/Logger';
import { BusinessError } from '@kit.BasicServicesKit';
import { common } from '@kit.AbilityKit';
import fileio from '@ohos.fileio';
import { tag } from '@kit.ConnectivityKit';
import { util } from '@kit.ArkTS';

const TAG: string = 'AudioEditTestApp_SongBox';

@Builder
export function SongBoxBuilder() {
  SongBox()
}

@Component
export struct SongBox {
  pageInfos: NavPathStack = new NavPathStack();
  @StorageLink('playList') playList: Map<string, SongInfo> = new Map(); // 播放列表，使用 Map 去重
  fileSize: number = 0;
  fileBuffer: ArrayBuffer = new ArrayBuffer(0);

  build() {
    NavDestination() {
      Column() {
        List({ initialIndex: 0 }) {
          if (this.playList.size != 0) {
            ForEach(Array.from(this.playList?.values()), (info: SongInfo) => {
              if (info.uri !== 'format' && info.uri !== 'resultPcmBuffer') {
                ListItem() {
                  Column() {
                    this.songItem(info);
                  }
                }
                .padding($r('app.float.padding_10'))
                .margin({ top: '10vp' })
                .backgroundColor(Color.Black)
                .border({
                  width: 3,
                  color: Color.Black,
                  radius: 10,
                  style: BorderStyle.Solid
                })
              }
            }, (info: SongInfo) => JSON.stringify(info))
          }
        }
      }
      .margin({ top: $r('app.float.margin_35') })
    }
    .onReady((context: NavDestinationContext) => {
      this.pageInfos = context.pathStack;
    })
    .title('音乐Box')
    .menus(this.NavigationMenus())
  }

  @Builder
  NavigationMenus() {
    Row() {
      Button($r('app.string.import'))
        .onClick((event: ClickEvent) => {
          importSongs(this.playList).then((songMap: Map<string, SongInfo>) => {
            this.playList = songMap;
          });
        })
        .width('60vp')
        .height('30vp')
    }
    .margin({ top: '10vp' })
  }

  /**
   * 歌曲栏
   * @param info  歌曲信息
   */
  @Builder
  songItem(info: SongInfo) {
    Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
      Image($r('app.media.icon_home_withdraw'))
        .width($r('app.float.width_50'))
        .height($r('app.float.height_50'))
        .margin({ right: $r('app.float.margin_35') })
      Text(info.songName).margin({ right: $r('app.float.margin_35') }).fontColor(Color.White)
    }.margin({ bottom: $r('app.float.margin_10') })

    Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
      Button($r('app.string.addSong')).onClick((event: ClickEvent) => {
        const result = this.getPcmFileBuffer(info.uri, info.songType);
        if (!result) {
          this.getUIContext().showAlertDialog({
            message: '只支持wav格式中44100/48000Hz的音频',
            confirm: {
              value: '确定',
              action: () => {
              }
            },
            cancel: () => {
              Logger.info(TAG, 'cancel');
            }
          });
        } else {
          // 添加音频，创建input和output节点，设置音频的文件格式
          this.pageInfos.pop(this.playList.get(info.uri));
        }
      })
      Button($r('app.string.delete'))
        .backgroundColor(Color.Red)
        .fontColor(Color.White)
        .onClick(() => {
          this.playList.delete(info.uri);
        })
    }
  }

  getPcmFileBuffer(uri: string, songType?: string): boolean {
    try {
      let path: string = new fileUri.FileUri(uri).path;
      Logger.info(TAG, `getPcmFileBuffer path: ${JSON.stringify(path)}`);
      let file = fs.openSync(path, fs.OpenMode.READ_ONLY | fs.OpenMode.CREATE);
      let fsStat = fs.statSync(path);
      Logger.info(TAG, `failSize : ${fsStat.size}`);
      let buffer = new ArrayBuffer(fsStat.size);
      let readOption: ReadOptions = {
        offset: 0, // 期望读取文件的位置。可选，默认从当前位置开始读
        length: fsStat.size // 每次期望读取数据的长度。可选，默认缓冲区长度
      }
      fs.readSync(file.fd, buffer, readOption);
      Logger.info(TAG, `getPcmFileBuffer buffer length: ${buffer.byteLength}`);
      let songName = this.playList.get(uri)?.songName ?? '';
      this.playList.set(uri, {
        uri: uri,
        songName: songName,
        songType: songType,
        wavBuffer: buffer as ArrayBuffer,
        fd: file.fd,
      });
      return true;
    } catch (e) {
      Logger.error(TAG, `getPcmFileBuffer error: ${JSON.stringify(e)}`);
      return false;
    }
  }
}