/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. All rights reserved.
 */

import audioNapi from 'libentry.so';
import { Logger } from '../utils/Logger';
import fs from '@ohos.file.fs';
import { common } from '@kit.AbilityKit';
import { picker } from '@kit.CoreFileKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { SongInfo } from '../utils/importSongs/InterfaceInfo';
import { MainTitleTextModifier, saveFileBuffer } from '../utils/ExportFile';
import { taskpool } from '@kit.ArkTS';
import { SymbolGlyphModifier } from '@kit.ArkUI';
import { getDateStringWithTimeStamp } from '../utils/Util';

const TAG: string = 'AudioEditTestApp_Export';

@Builder
export function ExportBuilder() {
  Export()
}

class RadioParam {
  public value: string = '';
  public groupName: string = '';
  public isChecked: boolean = false;
  public label: string = '';
  public onChange: VoidCallback = () => {
  }; // 点击单选框
}

@Component
export struct Export {
  pageInfos: NavPathStack = new NavPathStack();
  scroller: Scroller = new Scroller();
  // 主标题样式修改器
  @State mainTitleModifier: MainTitleTextModifier = new MainTitleTextModifier();
  private symbolModifier: SymbolGlyphModifier = new SymbolGlyphModifier($r('sys.symbol.chevron_backward')).fontColor([$r('app.color.index_title_text_color')]);
  @StorageLink('playList') playList: Map<string, SongInfo> = new Map();
  @StorageLink('songWaveList') songWaveList: Map<string, SongInfo> = new Map();
  private newAudioFileName: string = '';
  private audioFormate: string = '.wav';
  // 采样率
  private sampleRate: number = 8000;
  private sampleRates: number[] =
    [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000, 176400, 192000];
  // 通道
  private channels: number = 1;
  // 位深
  private bitsPerSample: number = 16;
  private bitsPerSamples: number[] = [8, 16, 24, 32];
  // 导出中
  @State isExported: boolean = false;

  aboutToAppear(): void {
    Logger.info(TAG, 'aboutToAppear');
  }

  build() {
    NavDestination() {
      Scroll(this.scroller) {
        Column() {
          Column() {
            Column() {
              Text('音频格式')
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.index_title_text_color'))
            }

            Column() {
              Row() {
                this.createRadio({
                  value: 'pcm',
                  groupName: 'radioGroupFormat',
                  isChecked: false,
                  label: 'pcm',
                  onChange: () => {
                    this.audioFormate = '.pcm';
                  }
                })
                this.createRadio({
                  value: 'wav',
                  groupName: 'radioGroupFormat',
                  isChecked: false,
                  label: 'wav',
                  onChange: () => {
                    this.audioFormate = '.wav';
                  }
                })
              }
              
            }.margin({ bottom: $r('app.float.margin_35') })

            Column() {
              Text('采样率')
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.index_title_text_color'))
            }

            Flex({
              direction: FlexDirection.Row,
              justifyContent: FlexAlign.SpaceBetween,
              alignItems: ItemAlign.Center,
              wrap: FlexWrap.Wrap
            }) {

              ForEach(this.sampleRates, (item: number) => {
                Row() {
                  this.createRadio({
                    value: `${item}`,
                    groupName: 'radioGroupRate',
                    isChecked: this.sampleRate === 16000 ? true : false,
                    label: `${item} Hz`,
                    onChange: () => {
                      this.sampleRate = item;
                    }
                  })
                }.width('100')
              })

            }.margin({ bottom: $r('app.float.margin_35') })

            Column() {
              Text('声道')
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.index_title_text_color'))
            }.margin({ bottom: $r('app.float.margin_10') })

            Flex({
              direction: FlexDirection.Row,
              justifyContent: FlexAlign.SpaceBetween,
              alignItems: ItemAlign.Center,
              wrap: FlexWrap.Wrap
            }) {
              Row() {
                this.createRadio({
                  value: '1',
                  groupName: 'radioGroupChannels',
                  isChecked: this.channels === 1 ? true : false,
                  label: '单声道',
                  onChange: () => {
                    this.channels = 1;
                  }
                })
              }

              Row() {
                this.createRadio({
                  value: '2',
                  groupName: 'radioGroupChannels',
                  isChecked: this.channels === 2 ? true : false,
                  label: '双声道',
                  onChange: () => {
                    this.channels = 2;
                  }
                })
              }
            }.margin({ bottom: $r('app.float.margin_35') })

            Column() {
              Text('位深')
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.index_title_text_color'))
            }.margin({ bottom: $r('app.float.margin_10') })

            Flex({
              direction: FlexDirection.Row,
              justifyContent: FlexAlign.SpaceBetween,
              alignItems: ItemAlign.Center,
              wrap: FlexWrap.Wrap
            }) {
              ForEach(this.bitsPerSamples, (item: number) => {
                Row() {
                  this.createRadio({
                    value: `${item}`,
                    groupName: 'radioGroupBitsPerSample',
                    isChecked: this.bitsPerSample === 32 ? true : false,
                    label: `${item}`,
                    onChange: () => {
                      this.bitsPerSample = item;
                    }
                  })
                }
              })
            }.margin({ bottom: $r('app.float.margin_10') })

            Column() {
              Text('输入保存的音频文件名')
                .margin({ right: '20vp' })
                .margin({ bottom: $r('app.float.margin_10') })
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.index_title_text_color'))
              TextInput({ text: this.newAudioFileName, placeholder: '输入保存的音频文件名' })
                .backgroundColor($r('app.color.index_title_text_color'))
                .onChange((value: string) => {
                  this.newAudioFileName = value;
                })
            }.margin({ top: $r('app.float.margin_10'), bottom: $r('app.float.margin_10') })
          }
          .alignItems(HorizontalAlign.Start)
          .margin({ top: $r('app.float.margin_35'), left: $r('app.float.margin_10'), right: $r('app.float.margin_10') })
          Column() {
            Button('保存音频格式').onClick((event: ClickEvent) => {
              Logger.info(TAG,
                `save sampleRate: ${this.sampleRate}, channels: ${this.channels}, bitsPerSample: ${this.bitsPerSample}`)
              audioNapi.setFormat(this.channels, this.sampleRate, this.bitsPerSample);
            })
              .backgroundColor($r('app.color.index_audio_clip_card'))
              .margin({bottom: $r('app.float.margin_10')})
            Button('确认导出')
              .enabled(!this.isExported)
              .backgroundColor($r('app.color.index_audio_clip_card'))
              .onClick(async () => {
                audioNapi.setFormat(this.channels, this.sampleRate, this.bitsPerSample);
                this.isExported = true;
                taskpool.execute(saveFileBuffer).then(async (value: object) => {
                  Logger.info(TAG, 'taskpool onClick start');
                  this.SaveBuffer(value as ArrayBuffer);
                  this.pageInfos.pop(this.channels);
                  this.isExported = false;
                  let tapValue: ArrayBuffer = audioNapi.getAudioOfTap();
                  if (tapValue.byteLength == 0) {
                    Logger.error(TAG, 'tapValue is blank');
                    return;
                  }
                  this.SaveTapBuffer(tapValue);
                });
                Logger.info(TAG, 'onClick end');
              })

          }.alignItems(HorizontalAlign.Center)
          .padding('10vp')
        }
      }
      .edgeEffect(EdgeEffect.None)
    }
    .title('导出音频', {mainTitleModifier: this.mainTitleModifier})
    .backButtonIcon(this.symbolModifier)
    .backgroundColor($r('app.color.main_backgroundColor'))
    .onReady((context: NavDestinationContext) => {
        this.pageInfos = context.pathStack;
    })
  }

  @Builder
  createRadio(param: RadioParam) {
    Radio({
      value: param.value,
      group: param.groupName,
      indicatorType: RadioIndicatorType.DOT,
    })
      .width(20)
      .height(20)
      .margin({ right: 10 })
      .checked(param.isChecked)
      .radioStyle({
        checkedBackgroundColor: $r('app.color.index_audio_clip_card'),
      })
      .onChange((isChecked: boolean) => {
        if (isChecked) {
          param.onChange();
        }
      })
    Text(param.label).margin({ right: '20vp' }).fontSize(14).fontColor($r('app.color.index_title_text_color'))
  }

  async SaveBuffer(value: ArrayBuffer) {
    try {
      Logger.info(TAG,
        `SaveBuffer start, audioFormate: ${this.audioFormate}, sampleRate: ${this.sampleRate}, channels: ${this.channels}, bitsPerSample: ${this.bitsPerSample}`);
      let pcmBuffer = value;
      Logger.info(TAG, `SaveBuffer pcmBuffer length : ${pcmBuffer.byteLength}`);
      if (pcmBuffer.byteLength === 0) {
        Logger.info(TAG, 'SaveBuffer error');
      }
      // picker选择器, 保存PCM文件
      let documentSaveOptions = new picker.DocumentSaveOptions();
      if (this.newAudioFileName === '') {
        this.newAudioFileName = getDateStringWithTimeStamp(new Date().getTime());
      } else {
        this.newAudioFileName = this.newAudioFileName + ' ' + getDateStringWithTimeStamp(new Date().getTime());
      }
      let newFileNames = `${this.newAudioFileName}${this.audioFormate}`;
      Logger.info(TAG, `SaveBuffer newFileNames is: ${newFileNames}`);
      documentSaveOptions.newFileNames = [newFileNames];
      let context = getContext() as common.Context; // 请确保 getContext(this) 返回结果为 UIAbilityContext
      let documentPicker = new picker.DocumentViewPicker(context);
      documentPicker.save(documentSaveOptions, (err: BusinessError, documentSaveResult: Array<string>) => {
        if (err) {
          Logger.error(TAG,
            `DocumentViewPicker.save failed with err, code is: ${err.code}, message is: ${err.message}`);
          return;
        }
        Logger.info(TAG,
          'DocumentViewPicker.save successfully, documentSaveResult uri: ' + JSON.stringify(documentSaveResult));
        let filePath = documentSaveResult;
        // documentSaveResult数组中只有一个路径
        let file = fs.openSync(documentSaveResult[0], fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
        Logger.info(TAG, `SaveBuffer filePath : ${filePath}`);
        if (this.audioFormate === '.wav') {
          // 处理wav
          let wavBuffer = this.writeWavFileHeader(pcmBuffer);
          let writeLen = fs.writeSync(file.fd, wavBuffer);
          Logger.info(TAG, `SaveWavBuffer writeLen : ${writeLen}`);
          fs.closeSync(file);
        } else {
          // 处理pcm
          let writeLen = fs.writeSync(file.fd, pcmBuffer);
          Logger.info(TAG, `SavePcmBuffer writeLen : ${writeLen}`);
          fs.closeSync(file);
        }
      })
    } catch (e) {
      Logger.error(TAG, `SaveBuffer catch: ${JSON.stringify(e)}`)
    }
  }

  async SaveTapBuffer(value: ArrayBuffer) {
    try {
      Logger.info(TAG,
        `SaveBuffer start, audioFormate: ${this.audioFormate}, sampleRate: ${this.sampleRate}, channels: ${this.channels}, bitsPerSample: ${this.bitsPerSample}`);
      let pcmBuffer = value;
      Logger.info(TAG, `SaveBuffer pcmBuffer length : ${pcmBuffer.byteLength}`);
      if (pcmBuffer.byteLength === 0) {
        Logger.info(TAG, 'SaveBuffer error');
      }
      // picker选择器, 保存PCM文件
      let documentSaveOptions = new picker.DocumentSaveOptions();
      let newFileNames = `tap_output${this.audioFormate}`;
      Logger.info(TAG, `SaveBuffer newFileNames is: ${newFileNames}`);
      documentSaveOptions.newFileNames = [newFileNames];
      let context = getContext() as common.Context; // 请确保 getContext(this) 返回结果为 UIAbilityContext
      let documentPicker = new picker.DocumentViewPicker(context);
      documentPicker.save(documentSaveOptions, (err: BusinessError, documentSaveResult: Array<string>) => {
        if (err) {
          Logger.error(TAG,
            `DocumentViewPicker.save failed with err, code is: ${err.code}, message is: ${err.message}`);
          return;
        }
        Logger.info(TAG,
          'DocumentViewPicker.save successfully, documentSaveResult uri: ' + JSON.stringify(documentSaveResult));
        let filePath = documentSaveResult;
        // documentSaveResult数组中只有一个路径
        let file = fs.openSync(documentSaveResult[0], fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
        Logger.info(TAG, `SaveBuffer filePath : ${filePath}`);
        if (this.audioFormate === '.wav') {
          // 处理wav
          let wavBuffer = this.writeWavFileHeader(pcmBuffer);
          let writeLen = fs.writeSync(file.fd, wavBuffer);
          Logger.info(TAG, `SaveWavBuffer writeLen : ${writeLen}`);
          fs.closeSync(file);
        } else {
          // 处理pcm
          let writeLen = fs.writeSync(file.fd, pcmBuffer);
          Logger.info(TAG, `SavePcmBuffer writeLen : ${writeLen}`);
          fs.closeSync(file);
        }
      })
    } catch (e) {
      Logger.error(TAG, `SaveBuffer catch: ${JSON.stringify(e)}`)
    }
  }

  writeString(dv: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      dv.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  concatArrayBuffer(wavHeaderBuffer: ArrayBuffer, pcmBuffer: ArrayBuffer) {
    const viewWavHeader = new Uint8Array(wavHeaderBuffer);
    const viewPcm = new Uint8Array(pcmBuffer);

    // 创建一个新的 ArrayBuffer, 大小为两个原始缓冲区之和
    const resultBuffer = new ArrayBuffer(viewWavHeader.byteLength + viewPcm.byteLength);
    const resultView = new Uint8Array(resultBuffer);

    // 将两个视图的数据复制到新的视图中
    resultView.set(viewWavHeader);
    resultView.set(viewPcm, viewWavHeader.length);

    return resultBuffer;
  }

  // 定义写 wav 头文件信息的方法
  writeWavFileHeader(pcmBuffer: ArrayBuffer): ArrayBuffer {
    Logger.info(TAG,
      `writeWavFileHeader sampleRate : ${this.sampleRate} --- channels : ${this.channels} --- bitsPerSample : ${this.bitsPerSample}`);
    const header = new ArrayBuffer(44);
    const dv = new DataView(header);
    // 写入RIFF快
    this.writeString(dv, 0, 'RIFF');
    let totalDataLen = pcmBuffer.byteLength + 44;
    dv.setUint32(4, totalDataLen, true);
    this.writeString(dv, 8, 'WAVE');
    // 写入 fmt 快
    this.writeString(dv, 12, 'fmt ');
    // fmt快大小
    dv.setUint32(16, 16, true);
    // 格式类别（1 - int， 3 - IEEE 浮点数）
    if (this.bitsPerSample === 32) {
      dv.setUint16(20, 3, true);
    } else {
      dv.setUint16(20, 1, true);
    }

    // 声道数
    dv.setUint16(22, this.channels, true);
    // 采样率
    dv.setUint32(24, this.sampleRate, true);
    // 比特率
    let byteRate = this.bitsPerSample * this.sampleRate * this.channels / 8;
    dv.setUint32(28, byteRate, true);
    // 每个采样点的字节数
    dv.setUint16(32, this.channels * this.bitsPerSample / 8, true);
    // 位深
    dv.setUint16(34, this.bitsPerSample, true);
    // 写入data 快
    this.writeString(dv, 36, 'data');
    // 数据快大小
    dv.setUint32(40, pcmBuffer.byteLength, true);

    let resultBuffer = this.concatArrayBuffer(header, pcmBuffer);
    return resultBuffer;
  }
}
