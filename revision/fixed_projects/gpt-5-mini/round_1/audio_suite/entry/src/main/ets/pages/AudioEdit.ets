
/*
* Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. All rights reserved.
*/

import {StyleConst} from '../utils/StyleConst';
import audioNapi from 'libentry.so';
import {Logger} from '../utils/Logger';
import {JSON, util} from '@kit.ArkTS';
import {SongInfo, NodeType, Node, ColorMap, SoundFiledType} from '../utils/importSongs/InterfaceInfo';

const TAG: string = 'AudioEditTestApp_AudioEdit';

@Builder
export function AudioEditBuilder() {
    AudioEdit()
}

@Component
export struct AudioEdit {
    pageInfos: NavPathStack = new NavPathStack();
    @StorageLink('playList') playList: Map<string, SongInfo> = new Map();
    @State effectBtnState: boolean = false;
    @State message: string = 'Hello World';
    @State fieldEqShow: boolean = false;
    @State separationShow: boolean = false;
    @State separatingShow: boolean = false;
    @State fieldEffectShow: boolean = false;
    @State envEffectShow: boolean = false;
    @State noiseReductionShow: boolean = false;
    @State voiceBeautifierShow: boolean = false;
    balanceModeScroller: Scroller = new Scroller();
    private balanceModeList: string[] = 
        ['默认', '民谣', '中国式', '古典', '舞曲', '爵士', '流行', 'R&B', '摇滚', '自定义'];
    @State equailizerMode: number = 0;
    @State balanceBandGains: number[] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    @State fileSize: number = 0;
    @State errorString: string = '';
    @State aissOptionMap: Map<string, number> = new Map();
    @State aissConfirmedMap: Map<string, boolean> = new Map();
    @State accompanimentChoiceState: boolean = false;
    @State fieldEffectNumMap: Map<string, SoundFiledType> = new Map();
    @State voiceBeautifierNumMap: Map<string, number> = new Map();
    @State environmentEffectNumMap: Map<string, number> = new Map();
    @StorageLink('songWaveList') songWaveList: Map<string, SongInfo> = new Map();
    @State value: boolean = false;
    @State pitchSong: string = '';
    @State aissUriToIdMap: Map<string, string | undefined> = new Map();
    @State fieldEffectIdMap: Map<string, string | undefined> = new Map();
    @State voiceBeautifierIdMap: Map<string, string | undefined> = new Map();
    @State environmentEffectUriToIdMap: Map<string, string | undefined> = new Map();
    @State noiseReductionUriToIdMap: Map<string, string | undefined> = new Map();
    @State noiseReductionProgress: number = 0;
    @State aissProgress: number = 0;
    @State outPutId: string = util.generateRandomUUID(true);
    @State mixerId: string = util.generateRandomUUID(true);
    @StorageLink('publicBuffer') pubBuffer: ArrayBuffer = new ArrayBuffer(0);
    @State openGridState: boolean = false;
    private equailizerId: string = '';
    
    aboutToAppear(): void {
        Logger.info(TAG, 'aboutToAppear');
        // 创建引擎， 创建管线
        try {
            let result = audioNapi.audioEditNodeInit(1);
            Logger.info(TAG, `audioEditNodeInit result: ${result}`);
        }catch (e) {
            Logger.info(TAG, `audioEditNodeInit error: ${JSON.stringify(e)}`);
        }
    }

    aboutToDisappear(): void {
        Logger.info(TAG, 'aboutToDisappear');
        // 销毁引擎， 销毁管线
        try {
            let result = audioNapi.audioEditDestory();
            Logger.info(TAG, `audioEditDestory result: ${result}`);
        }catch (e) {
            Logger.info(TAG, `audioEditDestory error: ${JSON.stringify(e)}`);
        }
    }

    build() {
        NavDestination() {
            Column() {
                Button('选择音乐').onClick(() => {
                    this.pageInfos.pushPath({
                        name: 'songBox', onPop: (popInfo: PopInfo) => {
                            if (this.songWaveList.size === 5) {
                                this.getUIContext().showAlertDialog({
                                    title: '提示',
                                    message: '音频同时处理的数量不能超过五个'
                                });
                                return;
                            }

                            const inputId: string = util.generateRandomUUID(true);

                            // 添加音频， 创建input和output节点， 设置音频的文件格式
                            this.selectedAudio(this.playList.get(popInfo.result['uri']) as SongInfo, inputId, this.outPutId,
                             this.mixerId);

                            const SongInfo: SongInfo = {
                                uri: popInfo.result['uri'],
                                songName: popInfo.result['songName'],
                                songType: popInfo.result['songType'],
                                nodes: [
                                    {id: inputId, type: NodeType.INPUT },
                                    {id: this.outPutId, type: NodeType.OUTPUT },
                                    {id: this.mixerId, type: NodeType.MIXER }
                                ]
                            }

                            this.songWaveList.set(inputId, SongInfo);
                            this.aissOptionMap.set(inputId, -1);
                            this.aissConfirmedMap.set(inputId, true);
                            this.fieldEffectNumMap.set(inputId, SoundFiledType.SOUND_FIELD_CLOSE);
                            this.fieldEffectIdMap.set(inputId, '');
                            this.voiceBeautifierIdMap.set(inputId, '');
                            this.voiceBeautifierNumMap.set(inputId, 0);
                            this.environmentEffectUriToIdMap.set(inputId, '');
                            this.environmentEffectNumMap.set(inputId, 0);
                            Logger.info(TAG, `input ID is , ${inputId}`);
                        }
                    })
                })
                this.SoundWave()
                Blank()
                    .layoutWeight(1)
                    .onClick(() => {
                        this.effectBtnState = false
                        this.openGridState = false
                    })
                this.AudioEditTool()
            }
        }
        .onReady((context: NavDestinationContext) => {
            this.pageInfos = context.pathStack;
        })
        .title('音频编辑')
        .menus(this.NavigationMenus())
        .height('100%')
        .width('100%')
    }

    @Builder
    NavigationMenus() {
        Row() {
            Button('导出')
                .width('60vp')
                .height('30vp')
                .onClick(() => {
                    this.pageInfos.pushPath({
                        name: 'export', onPop: (popInfo: PopInfo) => {
                            Logger.info(TAG, `loop deleteSong result and inputId`);
                            let flag = false;
                            this.songWaveList.forEach((songInfo, inputId) => {
                                let res = audioNapi.deleteSong(inputId);
                                Logger.info(TAG, `loop deleteSong result and inputId is ${res}, ${inputId}`);
                                if (res === 0) {
                                    this.songWaveList.delete(inputId);
                                } else {
                                    flag = true;
                                }
                            });
                            if (flag) {
                                this.getUIContext().showAlertDialog({
                                    title: '提示',
                                    message: '删除音频失败'
                                });
                            }
                        }
                    })
                })
        }
        .margin({ top: '10vp' })
    }

    @Builder
    AudioEditTool() {
        if (this.effectBtnState) {
            this.EffectButton()
        }
    }

    @Builder
    AudioSeparationDialog() {
        Column() {
            Row() {
                Column() {
                    Image($r('app.media.close_window'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    this.separationShow = false;
                    if (!this.aissConfirmedMap.get(this.pitchSong)) {
                        this.aissOptionMap.set(this.pitchSong, -1);
                    }
                })

                Column() {
                    Text('音源分离').fontSize(20)
                    Text('提取的音频类型将作于后续节点输入').fontSize(10).opacity(50)
                }
                .width('50%')

                Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.SpaceAround }) {
                    Button('提取').onClick(() => {
                        this.separationShow = false;
                        if (this.aissUriToIdMap.get(this.pitchSong) != undefined) {
                            this.getUIContext().showAlertDialog({
                                message: '如需重新添加，请先进行删除操作。\n请单击弹窗外任意处关闭弹窗',
                                autoCancel: true
                            });
                            return;
                        }
                        if (this.aissOptionMap.get(this.pitchSong) != -1) {
                            let oldId: string | undefined = this.aissUriToIdMap.get(this.pitchSong);
                            if (oldId == undefined) {
                                let uuid: string = util.generateRandomUUID(true);
                                let ret: number =
                                    audioNapi.addAudioSeparation(this.aissOptionMap.get(this.pitchSong), uuid, this.pitchSong);
                                if (ret != 0) {
                                    this.getUIContext().showAlertDialog({
                                        message: '添加分离节点失败\n请单击弹窗外任意处关闭弹窗',
                                        autoCancel: true
                                    });
                                    return;
                                }
                                this.aissUriToIdMap.set(this.pitchSong, uuid);
                                this.aissConfirmedMap.set(this.pitchSong, true);
                                let type: NodeType = 
                                    this.aissOptionMap.get(this.pitchSong) == 0 ? NodeType.VOICE : NodeType.ACCOMPANIMENT;
                                this.insertEffectLabel(uuid, type);
                                this.forceRenderByEffectBtn();
                            }
                            this.aissOptionMap.set(this.pitchSong, -1);
                        }
                    })
                        .height('30%')

                    Button('删除').onClick(() =>{
                        let oldId: string | undefined = this.aissUriToIdMap.get(this.pitchSong);
                        if (oldId == undefined) {
                            return;
                        }
                        this.separatingShow = false;
                        let ret: number = audioNapi.deleteAudioSeparation(oldId);
                        if (ret != 0) {
                            this.getUIContext().showAlertDialog({
                                message: '删除分离节点失败\n请单击弹窗外任意处关闭弹窗',
                                autoCancel: true
                            });
                            return;
                        }
                        let type: NodeType = 
                            this.aissOptionMap.get(this.pitchSong) == 0 ? NodeType.VOICE : NodeType.ACCOMPANIMENT;
                        this.deleteEffectLabel(type);
                        this.aissUriToIdMap.set(this.pitchSong, undefined);
                        this.forceRenderByEffectBtn();
                    })
                        .opacity(this.aissUriToIdMap.get(this.pitchSong) != undefined ? 1.0 : 0.2)
                        .height('30%')
                }
            }
            .height('40%')
            .alignItems(VerticalAlign.Center)

            Row() {
                Column() {
                    Row() {
                        Text('人声').fontSize(20).margin({right: 100})
                        if (this.separatingShow && this.aissProgress < 100) {
                            Progress({ value: this.aissProgress, total: 150, type: ProgressType.Ring }).width(30).height(30)
                        } else if (this.separationShow) {
                            Radio({ value: 'voice', group: 'separation' })
                                .onChange((isOn: boolean) => {
                                    if(isOn) {
                                        this.aissOptionMap.set(this.pitchSong, 0);
                                    }
                                })
                        }
                    }
                    .height('40%')

                    Row() {
                        Text('伴奏').fontSize(20).margin({right: 100})
                        if (this.separatingShow && this.aissProgress <= 100) {
                            Progress({ value: this.aissProgress, total: 150, type: ProgressType.Ring }).width(30).height(30)
                        } else if (this.separationShow) {
                            Radio({ value: 'company', group: 'separation' })
                                .onChange((isOn: boolean) => {
                                    if(isOn) {
                                        this.aissOptionMap.set(this.pitchSong, 1);
                                    }
                                })
                        }
                    }
                    .height('40%')
                }
            }
            .height('70%')
        }
    }

    @Builder
    FieldEffectDialog() {
        Column() {
            Row() {
                Column() {
                    Image($r('app.media.close_window'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    this.fieldEffectShow = false;
                })

                Column() {
                    Text('声场效果').fontSize(20)
                }
                .width('60%')

                Column() {
                    Image($r('app.media.checkmark'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    let fieldEffectId = this.fieldEffectIdMap.get(this.pitchSong);
                    if(fieldEffectId === '') {
                        let fieldId = util.generateRandomUUID(true);
                        //新增节点
                        if (this.fieldEffectNumMap.get(this.pitchSong) === SoundFiledType.SOUND_FIELD_CLOSE) {
                            this.fieldEffectShow = false;
                            return;
                        }
                        this.fieldEffectIdMap.set(this.pitchSong, fieldId);

                        let result = audioNapi.startFieldEffect(this.pitchSong, this.fieldEffectNumMap.get(this.pitchSong), fieldId);
                        if(result !== 0) {
                            // 效果节点设置失败
                            this.getUIContext().showAlertDialog({
                                message: '添加声场节点失败',
                                autoCancel: true
                            });
                            return;
                        }
                        this.playList.set('resultPcmBuffer', {
                            uri: 'resultPcmBuffer',
                            nodeId: fieldId
                        });
                        this.insertEffectLabel(fieldId, NodeType.FIELD);
                        this.fieldEffectShow = false;
                        this.forceRenderByEffectBtn();
                    } else {
                        //修改节点
                        let result = audioNapi.resetFieldEffect(this.pitchSong, this.fieldEffectNumMap.get(this.pitchSong), fieldEffectId);
                        if(result !== 0) {
                            this.getUIContext().showAlertDialog({
                                message: '设置声场节点失败',
                                autoCancel: true
                            });
                            return;
                        }
                        this.playList.set('resultPcmBuffer', {
                            uri: 'resultPcmBuffer',
                            nodeId: fieldEffectId
                        });
                        this.fieldEffectShow = false;
                        if (this.fieldEffectNumMap.get(this.pitchSong) == SoundFiledType.SOUND_FIELD_CLOSE) {
                            this.deleteEffectLabel(NodeType.FIELD);
                            this.fieldEffectIdMap.set(this.pitchSong, '');
                            this.forceRenderByEffectBtn();
                        }
                    }
                })
            }
            .height('30%')

            Row() {
                Column() {
                    Radio({ value: 'close', group: 'fieldEffect'})
                        .onChange(() => {
                            this.fieldEffectNumMap.set(this.pitchSong, SoundFiledType.SOUND_FIELD_CLOSE)
                        })
                    Text('无').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'front', group: 'fieldEffect'})
                        .onChange(() => {
                            this.fieldEffectNumMap.set(this.pitchSong, SoundFiledType.SOUND_FIELD_FRONT_FACING)
                        })
                    Text('前置').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'grand', group: 'fieldEffect'})
                        .onChange(() => {
                            this.fieldEffectNumMap.set(this.pitchSong, SoundFiledType.SOUND_FIELD_GRAND)
                        })
                    Text('宏大').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'near', group: 'fieldEffect'})
                        .onChange(() => {
                            this.fieldEffectNumMap.set(this.pitchSong, SoundFiledType.SOUND_FIELD_NEAR)
                        })
                    Text('聆听').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'broad', group: 'fieldEffect'})
                        .onChange(() => {
                            this.fieldEffectNumMap.set(this.pitchSong, SoundFiledType.SOUND_FIELD_WIDE)
                        })
                    Text('宽广').fontSize(20)
                }
                .width('20%')
            }
            .height('70%')
        }
    }

    @Builder
    EnvironmentEffectDialog() {
        Column() {
            Row() {
                Column() {
                    Image($r('app.media.close_window'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    this.envEffectShow = false;
                })

                Column() {
                    Text('环境效果').fontSize(20)
                }
                .width('60%')

                Column() {
                    Image($r('app.media.checkmark'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    this.envEffectShow = false;
                    if(this.environmentEffectNumMap.get(this.pitchSong) != undefined) {
                        let oldId: string | undefined = this.environmentEffectUriToIdMap.get(this.pitchSong);
                        if(oldId == '') {
                            if (this.environmentEffectNumMap.get(this.pitchSong) == 0) {
                                this.envEffectShow = false;
                                return;
                            }
                            this.envEffectShow = false;
                            let uuid: string = util.generateRandomUUID(true);
                            let ret: number = audioNapi.startEnvEffect(this.pitchSong, uuid, this.environmentEffectNumMap.get(this.pitchSong));
                            if(ret == 0) {
                                this.environmentEffectUriToIdMap.set(this.pitchSong, uuid);
                                this.insertEffectLabel(uuid, NodeType.ENV);
                                this.forceRenderByEffectBtn();
                            } else {
                                Logger.error(TAG, `startEnvEffect error number is ${ret}`);
                                this.getUIContext().showAlertDialog({
                                    message: '添加环境效果节点失败\n请单击弹窗外任意处关闭弹窗',
                                    autoCancel: true
                                });
                            }
                        } else {
                            let ret: number = audioNapi.resetEnvEffect(this.pitchSong, oldId, this.environmentEffectNumMap.get(this.pitchSong));
                            if(ret == 0) {
                                if (this.environmentEffectNumMap.get(this.pitchSong) == 0) {
                                    this.deleteEffectLabel(NodeType.ENV);
                                    this.environmentEffectUriToIdMap.set(this.pitchSong, undefined);
                                    this.forceRenderByEffectBtn();
                                }
                            } else {
                                Logger.error(TAG, `resetEnvEffect error number is ${ret}`);
                                this.getUIContext().showAlertDialog({
                                    message: '重置环境效果节点失败\n请单击弹窗外任意处关闭弹窗',
                                    autoCancel: true
                                });
                            }
                        }
                    }
                })
            }
            .height('30%')

            Row() {
                Column() {
                    Radio({ value: 'close', group: 'envEffect'})
                        .checked(this.environmentEffectNumMap.get(this.pitchSong) === 0)
                        .onChange(() => {
                            this.environmentEffectNumMap.set(this.pitchSong, 0)
                        })
                    Text('无').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'broadcast', group: 'envEffect'})
                        .checked(this.environmentEffectNumMap.get(this.pitchSong) === 1)
                        .onChange(() => {
                            this.environmentEffectNumMap.set(this.pitchSong, 1)
                        })
                    Text('广播').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'earpiece', group: 'envEffect'})
                        .checked(this.environmentEffectNumMap.get(this.pitchSong) === 2)
                        .onChange(() => {
                            this.environmentEffectNumMap.set(this.pitchSong, 2)
                        })
                    Text('听筒').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'underWater', group: 'envEffect'})
                        .checked(this.environmentEffectNumMap.get(this.pitchSong) === 3)
                        .onChange(() => {
                            this.environmentEffectNumMap.set(this.pitchSong, 3)
                        })
                    Text('水下').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'gramophone', group: 'envEffect'})
                        .checked(this.environmentEffectNumMap.get(this.pitchSong) === 4)
                        .onChange(() => {
                            this.environmentEffectNumMap.set(this.pitchSong, 4)
                        })
                    Text('留声机').fontSize(20)
                }
                .width('20%')
            }
            .height('70%')
        }
    }

    @Builder
    VoiceBeautifierDialog() {
        Column() {
            Row() {
                Column() {
                    Image($r('app.media.close_window'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    this.voiceBeautifierShow = false;
                })

                Column() {
                    Text('声音美化').fontSize(20)
                }
                .width('60%')

                Column() {
                    Image($r('app.media.checkmark'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    let voiceBeautifierId = this.voiceBeautifierIdMap.get(this.pitchSong);
                    if(voiceBeautifierId === '') {
                        let VBId = util.generateRandomUUID(true);
                        //选择普通则无美化效果
                        if (this.voiceBeautifierNumMap.get(this.pitchSong) === 0) {
                            this.voiceBeautifierShow = false;
                            return;
                        }
                        //新增节点
                        this.voiceBeautifierIdMap.set(this.pitchSong, VBId);
                        let ret = audioNapi.startVBEffect(this.pitchSong, this.voiceBeautifierNumMap.get(this.pitchSong), VBId);
                        if (ret !== 0) {
                          Logger.error(TAG, `addVoiceBeautify error number is ${ret}`);
                          this.getUIContext().showAlertDialog({
                            message: '添加美化节点失败\n请单击弹窗外任意处关闭弹窗',
                            autoCancel: true
                          })
                        } else {
                          this.playList.set('resultPcmBuffer', {
                            uri: 'resultPcmBuffer',
                            nodeId: VBId
                          });
                        }
                        this.insertEffectLabel(VBId, NodeType.VB);
                        this.voiceBeautifierShow = false;
                        this.forceRenderByEffectBtn();
                    } else {
                        //修改节点
                        let ret = audioNapi.resetVBEffect(this.pitchSong, this.voiceBeautifierNumMap.get(this.pitchSong), voiceBeautifierId);
                        if (ret !== 0) {
                          Logger.error(TAG, `resetVoiceBeautify error number is ${ret}`);
                          this.getUIContext().showAlertDialog({
                            message: '修改美化节点失败\n请单击弹窗外任意处关闭弹窗',
                            autoCancel: true
                          })
                        } else {
                          this.playList.set('resultPcmBuffer', {
                            uri: 'resultPcmBuffer',
                            nodeId: voiceBeautifierId
                          });
                        }
                        this.voiceBeautifierShow = false;
                        if (this.voiceBeautifierNumMap.get(this.pitchSong) == 0) {
                            this.deleteEffectLabel(NodeType.VB);
                            this.voiceBeautifierIdMap.set(this.pitchSong, '');
                            this.forceRenderByEffectBtn();
                        }
                    }
                })
            }
            .height('30%')

            Row() {
                Column() {
                    Radio({ value: 'normal', group: 'voiceBeautifier'})
                        .checked(this.voiceBeautifierNumMap.get(this.pitchSong) === 0)
                        .onChange(() => {
                            this.voiceBeautifierNumMap.set(this.pitchSong, 0)
                            Logger.info(TAG, `input ID is and VBmode , ${this.pitchSong}, ${this.voiceBeautifierNumMap.get(this.pitchSong)}`);
                        })
                    Text('普通').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'clear', group: 'voiceBeautifier'})
                        .checked(this.voiceBeautifierNumMap.get(this.pitchSong) === 1)
                        .onChange(() => {
                            this.voiceBeautifierNumMap.set(this.pitchSong, 1)
                            Logger.info(TAG, `input ID is and VBmode , ${this.pitchSong}, ${this.voiceBeautifierNumMap.get(this.pitchSong)}`);
                        })
                    Text('清澈').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'theatre', group: 'voiceBeautifier'})
                        .checked(this.voiceBeautifierNumMap.get(this.pitchSong) === 2)
                        .onChange(() => {
                            this.voiceBeautifierNumMap.set(this.pitchSong, 2)
                            Logger.info(TAG, `input ID is and VBmode , ${this.pitchSong}, ${this.voiceBeautifierNumMap.get(this.pitchSong)}`);
                        })
                    Text('剧场').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'cd', group: 'voiceBeautifier'})
                        .checked(this.voiceBeautifierNumMap.get(this.pitchSong) === 3)
                        .onChange(() => {
                            this.voiceBeautifierNumMap.set(this.pitchSong, 3)
                            Logger.info(TAG, `input ID is and VBmode , ${this.pitchSong}, ${this.voiceBeautifierNumMap.get(this.pitchSong)}`);
                        })
                    Text('CD音效').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'recordingStudio', group: 'voiceBeautifier'})
                        .checked(this.voiceBeautifierNumMap.get(this.pitchSong) === 4)
                        .onChange(() => {
                            this.voiceBeautifierNumMap.set(this.pitchSong, 4)
                            Logger.info(TAG, `input ID is and VBmode , ${this.pitchSong}, ${this.voiceBeautifierNumMap.get(this.pitchSong)}`);
                        })
                    Text('录音棚').fontSize(20)
                }
                .width('20%')
            }
            .height('70%')
        }
    }

    @Builder
    NoiseReductingDialog() {
        Row() {
            Column() {
                Row() {
                    Image($r('app.media.close_window'))
                        .width(20)
                        .height(20)
                        .onClick(() => {
                            this.noiseReductionShow = false;
                        })
                        .opacity(this.noiseReductionProgress > 0 ? 0 : 1)
                    Column() {
                    }.width('70%')
                }.height('30%')

                Row() {
                    Progress({ value: this.noiseReductionProgress, total: 100, type: ProgressType.Ring })
                }
            }.width('50%').height('100%')

            Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.SpaceAround }) {
                Row() {
                    Button('添加降噪').onClick(() =>{
                        if (this.noiseReductionUriToIdMap.get(this.pitchSong) != undefined) {
                            this.getUIContext().showAlertDialog({
                                message: '如需重新添加，请先进行删除操作。\n请单击弹窗外任意处关闭弹窗',
                                autoCancel: true
                            });
                            this.noiseReductionShow = false;
                            return;
                        }
                        let uuid: string = util.generateRandomUUID(true);
                        let ret: number = audioNapi.addNoiseReduction(uuid, this.pitchSong);
                        this.noiseReductionShow = false;
                        if (ret == 0) {
                            this.noiseReductionUriToIdMap.set(this.pitchSong, uuid);
                            this.insertEffectLabel(uuid, NodeType.NR);
                            this.forceRenderByEffectBtn();
                        } else {
                            Logger.error(TAG, `addNoiseReduction error number is ${ret}`);
                            this.getUIContext().showAlertDialog({
                                message: '添加降噪节点失败\n请单击弹窗外任意处关闭弹窗',
                                autoCancel: true
                            });
                        }
                    })
                }

                Row() {
                    Button('删除降噪').onClick(() =>{
                        let oldId: string | undefined = this.noiseReductionUriToIdMap.get(this.pitchSong);
                        if (oldId == undefined) {
                            return;
                        }

                        this.noiseReductionShow = false;
                        if (oldId != undefined) {
                            let ret: number = audioNapi.deleteNoiseReduction(oldId);
                            if (ret != 0) {
                                this.getUIContext().showAlertDialog({
                                    message: '删除降噪节点失败\n请单击弹窗外任意处关闭弹窗',
                                    autoCancel: true
                                });
                            }
                            this.deleteEffectLabel(NodeType.NR);
                            this.noiseReductionUriToIdMap.set(this.pitchSong, undefined);
                            this.forceRenderByEffectBtn();
                        }
                    })
                        .opacity(this.noiseReductionUriToIdMap.get(this.pitchSong) == undefined ? 0.2 : 1.0)
                }
            }.width('50%').height('100%')
        }
    }

    @Builder
    BalanceBuilder() {
        Column() {
            Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
                Column() {
                    Image($r('app.media.close_window'))
                        .width(20)
                        .height(20)
                }.onClick(() => {
                    this.fieldEqShow = false;
                })

                Text('均衡')
                Column() {
                    Image($r('app.media.checkmark'))
                        .width(20)
                        .height(20)
                }.onClick((event: ClickEvent) => {
                    if (!this.songWaveList.get(this.pitchSong)?.nodes?.find(label => label.type === NodeType.EQ)) {
                        const nodeId = util.generateRandomUUID(true);
                        this.songWaveList.get(this.pitchSong)?.nodes?.push({
                            id: nodeId,
                            type: NodeType.EQ,
                            color: ColorMap.get(NodeType.EQ)
                        });
                    }
                    this.forceRenderByEffectBtn();
                    if (this.equailizerId === '') {
                        this.equailizerId = util.generateRandomUUID(true);
                    }

                    if (this.equailizerMode !== 10) {
                        // 判断效果节点是否设置成功
                        let result = audioNapi.setEquailizerMode(this.equailizerMode, this.equailizerId, this.pitchSong);
                        if (this.equailizerMode === 1){
                            this.deleteEffectLabel(NodeType.EQ);
                        }
                        Logger.info(TAG, `setEquailizerMode result: ${result}, equailizerMode: ${this.equailizerMode}`);
                        if (result !== 0) {
                            // 效果节点设置失败
                            this.getUIContext().showAlertDialog({
                                message: '设置均衡器模式失败',
                                confirm: {
                                    value: '确定',
                                    action: () => {

                                    }
                                },
                                cancel: () =>{
                                    Logger.info(TAG, `cancel`);
                                }
                            });
                            return;
                        }
                        this.playList.set('resultPcmBuffer', {
                            uri: 'resultPcmBuffer',
                            nodeId: this.equailizerId
                        });
                    } else {
                        // 均衡器模式为自定义类型
                        let result =
                            audioNapi.setEqualizerFrequencyBandGains(this.balanceBandGains, this.equailizerId, this.pitchSong);
                        Logger.info(TAG, `setEqualizerFrequencyBandGains result: ${result}`);
                        if (result !== 0) {
                            // 效果节点设置失败
                            this.getUIContext().showAlertDialog({
                                message: '设置均衡器模式失败',
                                confirm: {
                                    value: '确定',
                                    action: () => {

                                    }
                                },
                                cancel: () =>{
                                    Logger.info(TAG, `cancel`);
                                }
                            });
                            return;
                        }
                        this.playList.set('resultPcmBuffer', {
                            uri: 'resultPcmBuffer',
                            nodeId: this.equailizerId
                        });
                    }
                    this.forceRenderByEffectBtn();
                })
            }.margin({ top:$r('app.float.margin_35')})

            Column() {
                Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems:ItemAlign.Center }) {
                    ForEach(this.balanceModeList.slice(0, 5), (item: string, index: number) => {
                        Button(item)
                            .align(Alignment.Center)
                            .fontColor(Color.White)
                            .borderRadius('20%')
                            .borderWidth('1vp')
                            .borderColor($r('app.color.index_audio_clip_card'))
                            .backgroundColor(this.equailizerMode === (index + 1) ? $r('app.color.index_audio_clip_card') :
                                Color.Gray)
                            .onClick(() => {
                                Logger.info(TAG, `index + 1 is ${index + 1}`);
                                this.equailizerMode = index + 1;
                            })
                    }, item => item)
                }

                Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems:ItemAlign.Center }) {
                    ForEach(this.balanceModeList.slice(5, 10), (item: string, index: number) => {
                        Button(item)
                            .align(Alignment.Center)
                            .fontColor(Color.White)
                            .borderRadius('20%')
                            .borderWidth('1vp')
                            .borderColor($r('app.color.index_audio_clip_card'))
                            .backgroundColor(this.equailizerMode === (index + 6) ? $r('app.color.index_audio_clip_card') :
                                Color.Gray)
                            .onClick(() => {
                                Logger.info(TAG, `index + 1 is ${index + 6}`);
                                this.equailizerMode = index + 6;
                            })
                    }, (item: string, index: number) => item)
                }
                .margin({ top: '20vp', bottom: '70vp'})
            }

            Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems:ItemAlign.Center }) {
                ForEach(this.balanceBandGains, (item: number, index: number) => {
                    Column() {
                        Text(`${item}`).fontSize(12)
                        Slider({
                            value: item,
                            step: 1,
                            style: SliderStyle.OutSet,
                            direction: Axis.Vertical,
                            min: -10,
                            max: 10,
                            reverse: true
                        })
                            .blockSize({ width: 30, height: 30 })
                            .blockBorderWidth(5)
                            .blockColor(Color.White)
                            .onChange((value: number, mode: SliderChangeMode) => {
                                this.balanceBandGains[index] = value;
                                // 均衡器模式为自定义模式
                                this.equailizerMode = 10;
                                this.balanceModeScroller.scrollToIndex(this.equailizerMode, true);
                            })
                    }.width('5%').height('200vp')
                }, (item: number, index: number) => item)
            }
            .height('200vp')
        }
        .padding({ left: $r('app.float.padding_20'), right: $r('app.float.padding_20')})
    }

    @Builder
    EffectButton() {
        if (!this.openGridState) {
            Row() {
                Column() {
                    Image($r('app.media.chevron_up'))
                        .width(20)
                        .height(20)
                        .onClick(() => {
                            this.openGridState = true;
                        })
                }
                .width('10%')

                List({space:'3vp'}) {
                    ListItem() {
                        this.DeleteButton()
                    }

                    ListItem() {
                        this.AudioSeparationButton()
                    }

                    ListItem() {
                        this.NoiseReductionButton()
                    }

                    ListItem() {
                        this.BalancerButton()
                    }

                    ListItem() {
                        this.FieldButton()
                    }

                    ListItem() {
                        this.EnvironmentButton()
                    }

                    ListItem() {
                        this.MixerButton()
                    }

                    ListItem() {
                        this.VoiceBeautifulButton()
                    }
                    
                }
                .width($r(StyleConst.NINETY_PERCENT_WIDTH))
                .height($r('app.float.height_50'))
                .margin({ right: $r('app.float.margin_35') })
                .listDirection(Axis.Horizontal)
                .alignListItem(ListItemAlign.Center)
                .scrollBar(BarState.Off)
            }
        } else {
            Image($r('app.media.chevron_down'))
                .width(20)
                .height(20)
                .margin({bottom:'5%'})
                .onClick(() => {
                    this.openGridState = false;
                })
            Grid() {
                GridItem() {
                    this.DeleteButton()
                }
                
                GridItem() {
                    this.AudioSeparationButton()
                }.margin({left: '10vp', right: '10vp'})

                GridItem() {
                    this.NoiseReductionButton()
                }

                GridItem() {
                    this.BalancerButton()
                }

                GridItem() {
                    this.FieldButton()
                }

                GridItem() {
                    this.EnvironmentButton()
                }

                GridItem() {
                    this.MixerButton()
                }

                GridItem() {
                    this.VoiceBeautifulButton()
                }

            }.height('25%').rowsGap('20%').columnsGap('3vp')
        }
    }

    @Builder
    VoiceBeautifulButton() {
        Column() {
            Button('声音美化', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
        }
        .onClick(() => {
            this.voiceBeautifierShow = true;
        })
        .bindSheet(this.voiceBeautifierShow, this.VoiceBeautifierDialog(), {
            height: '20%',
            backgroundColor: Color.White,
            showClose: false,
            onWillAppear: () => {
                Logger.info(TAG, `BindSheet onWillAppear`);
            },
            onAppear: () => {
                Logger.info(TAG, `BindSheet onAppear`);
            },
            onWillDisappear: () => {
                Logger.info(TAG, `BindSheet onWillDisappear`);
            },
            onDisappear: () => {
                this.voiceBeautifierShow = false;
                Logger.info(TAG, `BindSheet onDisappear`);
            }
        })
    }

    @Builder
    DeleteButton() {
        Column() {
            Button('删除', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
            .onClick(() => {
                if (this.songWaveList.size === 0) {
                    this.getUIContext().showAlertDialog({
                        message: '无音频',
                        autoCancel: true
                    });
                    return;
                }
                let res = audioNapi.deleteSong(this.pitchSong);
                Logger.info(TAG, `deleteSong result is ${res}`);
                if (res === 0) {
                    this.songWaveList.delete(this.pitchSong);
                } else {
                    this.getUIContext().showAlertDialog({
                        title: '提示',
                        message: '删除音频失败'
                    });
                }
            })
        }
    }

    @Builder
    NoiseReductionButton() {
        Column() {
            Button('降噪', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
            .onClick(() => {
                this.noiseReductionShow = true;
            })
        }
        .bindSheet(this.noiseReductionShow, this.NoiseReductingDialog(), {
            height: '20%',
            backgroundColor: Color.White,
            showClose: false,
            onWillAppear: () => {
            },
            onAppear: () => {
            },
            onWillDisappear: () => {
            },
            onDisappear: () => {
            },
            onWillDismiss: (DismissSheetAction: DismissSheetAction) => {}
        })
    }

    @Builder
    AudioSeparationButton() {
        Column() {
            Button('音源分离', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
            .onClick(() => {
                this.separationShow = true;
            })
            .bindSheet(this.separationShow, this.AudioSeparationDialog(), {
                height: '30%',
                backgroundColor: Color.White,
                showClose: false,
                onWillAppear: () => {
                },
                onAppear: () => {
                },
                onWillDisappear: () => {
                },
                onDisappear: () => {
                    this.separationShow = false;
                }
            })
        }
    }

    @Builder
    BalancerButton() {
        Column() {
            Button('均衡器', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
        }
        .onClick(() => {
            this.fieldEqShow = true;
        })
        .bindSheet(this.fieldEqShow, this.BalanceBuilder(), {
            height: SheetSize.MEDIUM,
            showClose: false,
            onWillAppear: () => {
                Logger.info(TAG, `BindSheet onWillAppear`);
            },
            onAppear: () => {
                Logger.info(TAG, `BindSheet onAppear`);
            },
            onWillDisappear: () => {
                Logger.info(TAG, `BindSheet onWillDisappear`);
            },
            onDisappear: () => {
                this.fieldEqShow = false;
                Logger.info(TAG, `BindSheet onDisappear`);
            }
        })
    }

    @Builder
    EnvironmentButton() {
        Column() {
            Button('环境', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
        }
        .onClick(() => {
            this.envEffectShow = true;
        })
        .bindSheet(this.envEffectShow, this.EnvironmentEffectDialog(), {
            height: '20%',
            backgroundColor: Color.White,
            showClose: false,
            onWillAppear: () => {
            },
            onAppear: () => {
            },
            onWillDisappear: () => {
            },
            onDisappear: () => {
                this.envEffectShow = false;
            }
        })
    }

    @Builder
    FieldButton() {
        Column() {
            Button('声场', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
        }
        .onClick(() => {
            this.fieldEffectShow = true;
        })
        .bindSheet(this.fieldEffectShow, this.FieldEffectDialog(), {
            height: '20%',
            backgroundColor: Color.White,
            showClose: false,
            onWillAppear: () => {
                Logger.info(TAG, `BindSheet onWillAppear`);
            },
            onAppear: () => {
                Logger.info(TAG, `BindSheet onAppear`);
            },
            onWillDisappear: () => {
                Logger.info(TAG, `BindSheet onWillDisappear`);
            },
            onDisappear: () => {
                this.fieldEffectShow = false;
                Logger.info(TAG, `BindSheet onDisappear`);
            }
        })
    }

    @Builder
    MixerButton() {
        Column() {
            Button('混音', { type: ButtonType.Normal, stateEffect: true }).backgroundColor(0x317aff).borderRadius(8).width(80)
        }
    }

    @Builder
    SoundWave() {
        List({ initialIndex: 0 }) {
            if (this.songWaveList.size != 0) {
                ForEach(Array.from(this.songWaveList.keys()), (key: string) =>{
                    if (this.pitchSong === key && this.effectBtnState) {
                        ListItem() {
                            this.songItem(key, this.songWaveList.get(key) as SongInfo);
                        }
                        .padding($r('app.float.padding_10'))
                        .margin({ top: '10vp' })
                        .backgroundColor(Color.Black)
                        .border({
                            width: 3,
                            color: 0x317AF7,
                            radius: 10,
                            style: BorderStyle.Solid
                        })
                    } else {
                        ListItem() {
                            this.songItem(key, this.songWaveList.get(key) as SongInfo);
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

    /**
     * 歌曲栏
     * @param info 歌曲信息
     */
    @Builder
    songItem(nodeId: string, info: SongInfo) {
        Column() {
            Row() {
                Image($r('app.media.icon_home_withdraw'))
                    .width('app.float.width_50')
                    .height($r('app.float.height_50'))
                    .margin({ right: $r('app.float.margin_35') })
                Text(info.songName).margin({ right: 30 }).fontColor(Color.White)
            }
            .justifyContent(FlexAlign.SpaceBetween)

            Row() {
                List() {
                    ForEach(this.songWaveList.get(nodeId)?.nodes ?? [], (item: Node) => {
                        ListItem() {
                            Text(item.type)
                                .fontSize(14)
                                .fontColor('#ffffff')
                        }
                        .visibility(!(item.type === NodeType.INPUT || item.type === NodeType.OUTPUT ||
                            item.type === NodeType.MIXER) ? Visibility.Visible : Visibility.None)
                        .borderRadius(20)
                        .margin($r('app.float.margin_5'))
                        .padding($r('app.float.margin_5'))
                        .backgroundColor(item.color)
                    }, (item: Node) => JSON.stringify(item))
                        .onMove((from: number, to: number) => {
                            Logger.error(TAG, `onMove`);
                            let tmp = this.songWaveList.get(nodeId)?.nodes?.splice(from, 1);
                            if (tmp) {
                                this.songWaveList.get(nodeId)?.nodes?.splice(to, 0, tmp[0]);
                            }
                        })
                }
                .width('100%')
                .listDirection(Axis.Horizontal)
                .alignListItem(ListItemAlign.Center)
                .focusWrapMode(FocusWrapMode.WRAP_WITH_ARROW)
                .scrollBar(BarState.Off)
            }
        }
        .margin({ bottom: $r('app.float.margin_0') }).width('100%').onClick(() => {
            this.pitchSong = nodeId;
            this.effectBtnState = true;
        })
    }

    selectedAudio(info: SongInfo, inputId: string, outPutId: string, mixerId: string) {
        Logger.info(TAG, 
            `songType: ${info.songType},  wavBuffer length : ${info.wavBuffer?.byteLength}`);
        if (info.songType === 'wav' && info.wavBuffer) {
            // 调用napi接口
            try {
                let result =
                    audioNapi.audioInAndOutInit(inputId, outPutId, mixerId, info.fd, info.wavBuffer.byteLength);
                if (result !== 0) {
                    this.getUIContext().showAlertDialog({
                        title: '提示',
                        message: `error: ${result}`
                    });
                }
                audioNapi.audioRendererInit();
            }catch(e) {
                Logger.info(TAG, `err: ${e}`);
            }
        } else {
            Logger.info(TAG, `get file failed`);
        }
    }

    // 强制渲染的cheat方式
    forceRenderByEffectBtn() {
        this.effectBtnState = false;
    }

    insertEffectLabel(effectNodeId: string, type: NodeType): boolean {
        if (!this.songWaveList.get(this.pitchSong)?.nodes?.find(label => label.type === type)) {
            this.songWaveList.get(this.pitchSong)?.nodes?.push({
                id: effectNodeId,
                type: type,
                color: ColorMap.get(type)
            });
        }
        return true;
    }

    deleteEffectLabel(type: NodeType): boolean {
        let index: number | undefined = this.songWaveList.get(this.pitchSong)?.nodes?.findIndex(node => node.type === type)
        if (index != undefined) {
            this.songWaveList.get(this.pitchSong)?.nodes?.splice(index, 1);
        }
        return true;
    }

    resetEffectLabel(index: number, effectNodeId: string, destType: NodeType): boolean {
        let nodeArray: Node[] | undefined = this.songWaveList.get(this.pitchSong)?.nodes;
        if (index != undefined && nodeArray != undefined) {
            nodeArray[index] = { id: effectNodeId, type: destType, color: ColorMap.get(destType) };
        }
        return true;
    }
}