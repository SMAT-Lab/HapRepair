
/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

import { MainTitleTextModifier, saveFileBuffer } from '../../utils/ExportFile'
import { SymbolGlyphModifier } from '@kit.ArkUI';
import { SongInfo } from '../../utils/importSongs/InterfaceInfo';
import audioNapi from 'libentry.so';

class RadioParam {
    public value: string = '';
    public groupName: string = '';
    public isChecked: boolean = false;
    public label: string = '';
    public onChange: VoidCallback = () => {
    }; // 点击单选框
};

@CustomDialog
export struct SetFormatDialog {
    controller?: CustomDialogController;
    //判断无音频，保存格式按钮置灰
    @Prop nodeList: Map<string, SongInfo> = new Map();
    // 主标题样式修改器
    mainTitleModifier: MainTitleTextModifier = new MainTitleTextModifier();
    private symbolModifer: SymbolGlyphModifier =
        new SymbolGlyphModifier($r('sys.symbol.chevron_backward')).fontColor([$r('app.color.index_title_text_color')])
    @StorageLink('playList') playList: Map<string, SongInfo> = new Map();
    @StorageLink('songWaveList') songWaveList: Map<string, SongInfo> = new Map();
    @Link newAudioFileName: string;
    private audioFormate: string = '.wav';
    // 采样率
    @Link sampleRate: number;
    private sampleRates: number[] =
        [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000, 176400, 192000];
    // 通道
    @Link channels: number;
    // 位深
    @Link bitsPerSample: number;
    private bitsPerSamples: number[] = [8, 16, 24, 32];
    // 导出中
    isExported: boolean = false;
    scroller: Scroller = new Scroller();
    cancel: () => void = () => {
    }
    confirm: () => void = () => {
    }

    build() {
        Scroll(this.scroller) {
            Column() {
                Column() {
                    Column() {
                        Text('音频格式')
                            .fontWeight(FontWeight.Bold)
                            .fontColor($r('app.color.index_title_text_color'))
                    }
                    .margin({ top: $r('app.float.margin_10') })

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

                    }.margin({ bottom: $r('app.float.margin_10') })

                    Text('采样率（混音输入最高采样率）')
                        .fontWeight(FontWeight.Bold)
                    // .fontColor($r('app.color.index_title_text_color))

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
                                    isChecked: this.sampleRate === item ? true : false,
                                    label: `${item} Hz`,
                                    onChange: () => {
                                        this.sampleRate = item;
                                    }
                                })
                            }.width('100')
                        }, (item: number) => `${item}`)
                    
                    }.margin({ bottom: $r('app.float.margin_10') })

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
                    }.margin({ bottom: $r('app.float.margin_10') })

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
                                    isChecked: this.bitsPerSample === item ? true : false,
                                    label: `${item}`,
                                    onChange: () => {
                                        this.bitsPerSample = item;
                                    }
                                })
                            }
                        }, (item: number) => item)
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
                    }.margin({ bottom: $r('app.float.margin_10') })

                }
                .alignItems(HorizontalAlign.Start)
                .margin({
                    bottom: $r('app.float.margin_10'),
                    left: $r('app.float.margin_10'),
                    right: $r('app.float.margin_10')
                })

                Column() {
                    Button($r('app.string.save_audio_format'))
                        .opacity(this.nodeList.size === 0 ? 0.2 : 1.0)
                        .backgroundColor($r('app.color.index_audio_clip_card'))
                        .onClick(() => {
                            if (this.nodeList.size === 0) {
                                this.getUIContext().showAlertDialog({
                                    message: '请先添加音频',
                                    autoCancel: true
                                });
                                return;
                            }
                            audioNapi.setFormat(this.channels, this.sampleRate, this.bitsPerSample);
                            this.controller?.close();
                        })
                }
                .alignItems(HorizontalAlign.Center)
                .margin({ bottom: $r('app.float.margin_10') })
            }
            .padding({ top: $r('app.float.margin_10'), left: $r('app.float.margin_10'), right: $r('app.float.margin_10') })
            .backgroundColor($r('app.color.main_backgroundColor'))
        }
        .edgeEffect(EdgeEffect.None)

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
        Text(param.label).margin({ right: '20vp' }).fontSize(14)
            .fontColor($r('app.color.index_title_text_color'))
    }
}