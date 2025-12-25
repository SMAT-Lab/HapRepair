/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
 
import {
    Node,
    NodeType,
    ColorMap,
    SongInfo,
    SoundFiledType,
    EnvironmentType,
    VoiceBeautifierType,
    SoundFiledMap,
    EnvironmentMap,
    VoiceBeautifierMap,
    AudioSeparationType,
    AudioSeparationMap,
    NumberToNodeTypeMap
} from "../../utils/importSongs/InterfaceInfo";
import { JSON, util } from '@kit.ArkTS';
import { Logger } from '../../utils/Logger';
import { AlertDialog } from '@kit.ArkUI';
import audioNapi from 'libentry.so';

const TAG: string = 'AudioEditTestApp_CheckNodeDialog';

// 均衡
@Component
struct EquailizerComponent {
    // 均衡频带
    @Link balanceBandGains: number[];
    onCancel?: () => void;
    onAgree?: () => void;

    build() {
        Column() {
            Text(`${NodeType.EQ}`)
                .margin({ bottom: $r('app.float.margin_10') })
            Flex({
                direction: FlexDirection.Row,
                justifyContent: FlexAlign.SpaceBetween,
                alignItems: ItemAlign.Center
            }) {
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
                            // .showSteps(true)
                            .onChange((value: number, mode: SliderChangeMode) => {
                                this.balanceBandGains[index] = value;
                            })
                    }.width('5%').height('200vp')
                })
            }

            Flex({
                direction: FlexDirection.Row,
                justifyContent: FlexAlign.SpaceBetween,
                alignItems: ItemAlign.Center
            }) {
                Button('取消')
                    .onClick(() => {
                        if (this.onCancel) {
                            this.onCancel()
                        }
                    })
                Button('确定')
                    .onClick(() => {
                        if (this.onAgree) {
                            this.onAgree();
                        }
                    })
            }.margin({ top: $r('app.float.margin_10') })
        }.padding($r('app.float.padding_10'))
    }
}

// 声场
@Component
struct FieldComponent {
    @Link selectedSoundFiled: SoundFiledType;
    onCancel?: () => void;
    onAgree?: () => void;

    build() {
        Column() {
            Text(`${NodeType.FIELD}`)
                .margin({ bottom: $r('app.float.margin_10') })
            Row() {
                Column() {
                    Radio({ value: 'close', group: 'fieldEffect' })
                        .checked(this.selectedSoundFiled === SoundFiledType.SOUND_FIELD_CLOSE)
                        .onChange(() => {
                            // tongjian OH_AudioSuite_Result OH_AudioSuiteEngine_BypassEffectNode(OH_AudioNode* audioNode, bool bypass);
                            this.selectedSoundFiled = SoundFiledType.SOUND_FIELD_CLOSE;
                        })
                    Text('无').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'front', group: 'fieldEffect' })
                        .checked(this.selectedSoundFiled === SoundFiledType.SOUND_FIELD_FRONT_FACING)
                        .onChange(() => {
                            this.selectedSoundFiled = SoundFiledType.SOUND_FIELD_FRONT_FACING;
                        })
                    Text('前置').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'grand', group: 'fieldEffect' })
                        .checked(this.selectedSoundFiled === SoundFiledType.SOUND_FIELD_GRAND)
                        .onChange(() => {
                            this.selectedSoundFiled = SoundFiledType.SOUND_FIELD_GRAND;
                        })
                    Text('宏大').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'near', group: 'fieldEffect' })
                        .checked(this.selectedSoundFiled === SoundFiledType.SOUND_FIELD_NEAR)
                        .onChange(() => {
                            this.selectedSoundFiled = SoundFiledType.SOUND_FIELD_NEAR;
                        })
                    Text('聆听').fontSize(20)
                }
                .width('20%')

                Column() {

                    Radio({ value: 'broad', group: 'fieldEffect' })
                        .checked(this.selectedSoundFiled === SoundFiledType.SOUND_FIELD_WIDE)
                        .onChange(() => {
                            this.selectedSoundFiled = SoundFiledType.SOUND_FIELD_WIDE;
                        })
                    Text('宽广').fontSize(20)
                }
                .width('20%')
            }

            Flex({
                direction: FlexDirection.Row,
                justifyContent: FlexAlign.SpaceBetween,
                alignItems: ItemAlign.Center
            }) {
                Button('取消')
                    .onClick(() => {
                        if (this.onCancel) {
                            this.onCancel()
                        }
                    })
                Button('确定')
                    .onClick(() => {
                        if (this.onAgree) {
                            this.onAgree();
                        }
                    })
            }.margin({ top: $r('app.float.margin_10') })
        }
        .padding($r('app.float.padding_10'))
    }
}

// 环境
@Component
struct EnvironmentComponent {
    @Link selectedEnvironment: EnvironmentType;
    onCancel?: () => void;
    onAgree?: () => void;

    build() {
        Column() {
            Text(`${NodeType.ENV}`)
                .margin({ bottom: $r('app.float.margin_10') })
            Row() {
                Column() {
                    Radio({ value: 'close', group: 'envEffect' })
                        .checked(this.selectedEnvironment === EnvironmentType.ENVIRONMENT_TYPE_CLOSE)
                        .onChange(() => {
                            this.selectedEnvironment = EnvironmentType.ENVIRONMENT_TYPE_CLOSE;
                        })
                    Text('无').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'broadcast', group: 'envEffect' })
                        .checked(this.selectedEnvironment === EnvironmentType.ENVIRONMENT_TYPE_BROADCAST)
                        .onChange(() => {
                            this.selectedEnvironment = EnvironmentType.ENVIRONMENT_TYPE_BROADCAST;
                        })
                    Text('广播').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'earpiece', group: 'envEffect' })
                        .checked(this.selectedEnvironment === EnvironmentType.ENVIRONMENT_TYPE_EARPIECE)
                        .onChange(() => {
                            this.selectedEnvironment = EnvironmentType.ENVIRONMENT_TYPE_EARPIECE;
                        })
                    Text('听筒').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'underWater', group: 'envEffect' })
                        .checked(this.selectedEnvironment === EnvironmentType.ENVIRONMENT_TYPE_UNDERWATER)
                        .onChange(() => {
                            this.selectedEnvironment = EnvironmentType.ENVIRONMENT_TYPE_UNDERWATER;
                        })
                    Text('水下').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'gramophone', group: 'envEffect' })
                        .checked(this.selectedEnvironment === EnvironmentType.ENVIRONMENT_TYPE_GRAMOPHONE)
                        .onChange(() => {
                            this.selectedEnvironment = EnvironmentType.ENVIRONMENT_TYPE_GRAMOPHONE;
                        })
                    Text('留声机').fontSize(20)
                }
                .width('20%')
            }

            Flex({
                direction: FlexDirection.Row,
                justifyContent: FlexAlign.SpaceBetween,
                alignItems: ItemAlign.Center
            }) {
                Button('取消')
                    .onClick(() => {
                        if (this.onCancel) {
                            this.onCancel()
                        }
                    })
                Button('确定')
                    .onClick(() => {
                        if (this.onAgree) {
                            this.onAgree();
                        }
                    })
            }.margin({ top: $r('app.float.margin_10') })
        }
        .padding($r('app.float.padding_10'))
    }
}

// 降噪
@Component
struct NoiseReductionComponent {
    onCancel?: () => void;
    onAgree?: () => void;

    build() {
        Column() {
            Text(`${NodeType.NR}`)
                .margin({ bottom: $r('app.float.margin_10') })
            Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
                Button('添加降噪')
                    .onClick(() => {
                        if (this.onAgree) {
                            this.onAgree();
                        }
                    })
            }.margin({ top: $r('app.float.margin_10') })
        }
        .padding($r('app.float.padding_10'))
    }
}

// 美化
@Component
struct VoiceBeautifierComponent {
    @Link selectedVoiceBeautifier: VoiceBeautifierType;
    onCancel?: () => void;
    onAgree?: () => void;

    build() {
        Column() {
            Text(`${NodeType.VB}`)
                .margin({ bottom: $r('app.float.margin_10') })
            Row() {

                Column() {
                    Radio({
                        value: 'clear',
                        group: 'voiceBeautifier'
                    })
                        .checked(this.selectedVoiceBeautifier === VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_CLEAR)
                        .onChange(() => {
                            this.selectedVoiceBeautifier = VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_CLEAR;
                        })
                    Text('清澈').fontSize(20)
                }
                .width('20%')


                Column() {
                    Radio({ value: 'theatre', group: 'voiceBeautifier' })
                        .checked(this.selectedVoiceBeautifier === VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_THEATRE)
                        .onChange(() => {
                            this.selectedVoiceBeautifier = VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_THEATRE;
                        })
                    Text('剧场').fontSize(20)
                }
                .width('20%')

                Column() {
                    Radio({ value: 'cd', group: 'voiceBeautifier' })
                        .checked(this.selectedVoiceBeautifier === VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_CD)
                        .onChange(() => {
                            this.selectedVoiceBeautifier = VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_CD;
                        })
                    Text('CD音效').fontSize(20)
                }
                .width('20%')

                Column() {

                    Radio({ value: 'recordingStudio', group: 'voiceBeautifier' })
                        .checked(this.selectedVoiceBeautifier === VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_STUDIO)
                        .onChange(() => {
                            this.selectedVoiceBeautifier = VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_STUDIO;
                        })
                    Text('录音棚').fontSize(20)
                }
                .width('20%')
            }

            Flex({
                direction: FlexDirection.Row,
                justifyContent: FlexAlign.SpaceBetween,
                alignItems: ItemAlign.Center
            }) {
                Button('取消')
                    .onClick(() => {
                        if (this.onCancel) {
                            this.onCancel()
                        }
                    })
                Button('确定')
                    .onClick(() => {
                        if (this.onAgree) {
                            this.onAgree();
                        }
                    })
            }.margin({ top: $r('app.float.margin_10') })
        }
        .padding($r('app.float.padding_10'))
    }
}

// 音源分离
@Component
struct AudioSeparationComponent {
    @Link selectedAudioSeparation: AudioSeparationType;
    onCancel?: () => void;
    onAgree?: () => void;
    // separatingShow 未发生变化，改为普通变量
    separatingShow: boolean = false;
    @State aissProgress: number = 0;

    build() {
        Column() {
            Row() {
                Column() {
                    Image($r('app.media.close_window'))
                        .width(20)
                        .height(20)
                }
                .width('20%')
                .onClick(() => {
                    if (this.onCancel) {
                        this.onCancel();
                    }
                })

                Column() {
                    Text('音源分离').fontSize(20)
                    Text('提取的音频类型将作于后续节点输入').fontSize(10).opacity(50)
                }
                .width('50%')

                Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.SpaceAround }) {
                    Button('提取').onClick(() => {
                        if (this.onAgree) {
                            this.onAgree();
                        }
                    })
                        .height($r('app.float.height_50'))
                }
            }
            .height($r('app.float.height_100'))
            .alignItems(VerticalAlign.Center)

            Row() {
                Column() {
                    Row() {
                        Text('人声').fontSize(20).margin({ right: 100 })
                        if (this.separatingShow && this.aissProgress < 100) {
                            Progress({ value: this.aissProgress, total: 150, type: ProgressType.Ring })
                                .width(30)
                                .height(30)
                        } else {
                            Radio({ value: 'human', group: 'separation' })
                                .checked(this.selectedAudioSeparation ===
                                AudioSeparationType.AUDIO_NODE_HUMAN_SOUND_OUTPORT_TYPE)
                                .onChange((isOn: boolean) => {
                                    this.selectedAudioSeparation =
                                        AudioSeparationType.AUDIO_NODE_HUMAN_SOUND_OUTPORT_TYPE
                                })
                        }
                    }
                    .height($r('app.float.height_50'))

                    Row() {
                        Text('伴奏 ').fontSize(20).margin({ right: 100 })
                        if (this.separatingShow && this.aissProgress <= 100) {
                            Progress({ value: this.aissProgress, total: 150, type: ProgressType.Ring })
                                .width(30)
                                .height(30)
                        } else {
                            Radio({ value: 'backGround', group: 'separation' })
                                .checked(this.selectedAudioSeparation ===
                                AudioSeparationType.AUDIO_NODE_BACKGROUND_SOUND_OUTPORT_TYPE)
                                .onChange((isOn: boolean) => {
                                    if (isOn) {
                                        this.selectedAudioSeparation =
                                            AudioSeparationType.AUDIO_NODE_BACKGROUND_SOUND_OUTPORT_TYPE
                                    }
                                })
                        }
                    }
                    .height($r('app.float.height_50'))
                }

            }
            .height($r('app.float.height_100'))

        }
    }
}

@CustomDialog
struct EffectNodeDialog {
    @Prop selectedNode: string = '';
    @Prop selectedEffectNode: NodeType;
    @Prop currentInputId: string = '';
    @Prop isLeft: boolean;
    @Link mixerList: Array<Node>;
    @Link inputList: SongInfo[];
    @Link nodeList: Map<string, SongInfo>;
    // 均衡频带
    @State balanceBandGains: number[] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    //声场数据，默认是无
    @State selectedSoundFiled: SoundFiledType = SoundFiledType.SOUND_FIELD_CLOSE;
    //环境数据，默认是无
    @State selectedEnvironment: EnvironmentType = EnvironmentType.ENVIRONMENT_TYPE_CLOSE;
    //美化数据，默认是普通
    @State selectedVoiceBeautifier: VoiceBeautifierType = VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_NORMAL;
    //降噪数据，默认是无
    selectedNoiseReduction: number = -1;
    //音源分离，默认为人声
    @State selectedAudioSeparation: AudioSeparationType = AudioSeparationType.AUDIO_NODE_HUMAN_SOUND_OUTPORT_TYPE;
    controllerTwo?: CustomDialogController;

    build() {
        Column() {
            if (this.selectedEffectNode === NodeType.EQ) {
                EquailizerComponent({
                    balanceBandGains: this.balanceBandGains,
                    onCancel: () => {
                        this.controllerTwo?.close();
                    },
                    onAgree: () => {
                        let equailizerId = util.generateRandomUUID(true);
                        let result =
                            audioNapi.setEqualizerFrequencyBandGains(this.balanceBandGains, equailizerId,
                                this.currentInputId, this.selectedNode);
                        if (result !== 0) {
                            Logger.error(TAG, `setEqualizerFrequencyBandGains result: ${result}`);
                            this.getUIContext().showAlertDialog({
                                message: '添加均衡节点失败',
                                autoCancel: true
                            });
                            return;
                        }

                        //添加效果节点到混音台---数据处理
                        let node: Node = {
                            id: equailizerId,
                            type: this.selectedEffectNode,
                            color: ColorMap.get(this.selectedEffectNode),
                            balanceBandGains: this.balanceBandGains
                        }
                        this.isLeft ? this.addEffectNodeToInputList(node) : this.addEffectNodeToMixerList(node);

                        this.controllerTwo?.close();
                    }
                })
            } else if (this.selectedEffectNode === NodeType.FIELD) {
                FieldComponent({
                    selectedSoundFiled: this.selectedSoundFiled,
                    onCancel: () => {
                        this.controllerTwo?.close();
                    },
                    onAgree: () => {
                        // 调用napi的底层效果节点
                        const nodeId = util.generateRandomUUID(true);
                        let result = audioNapi.startFieldEffect(this.currentInputId, this.selectedSoundFiled, nodeId,
                            this.selectedNode);
                        if (result !== 0) {
                            this.getUIContext().showAlertDialog({
                                message: '添加声场节点失败',
                                autoCancel: true
                            });
                            return;
                        }
                        //添加效果节点到混音台---数据处理
                        let node: Node = {
                            id: nodeId,
                            type: this.selectedEffectNode,
                            color: ColorMap.get(this.selectedEffectNode),
                            soundFiledType: this.selectedSoundFiled
                        }
                        this.isLeft ? this.addEffectNodeToInputList(node) : this.addEffectNodeToMixerList(node);

                        this.controllerTwo?.close();
                    }
                })
            } else if (this.selectedEffectNode === NodeType.ENV) {
                EnvironmentComponent({
                    selectedEnvironment: this.selectedEnvironment,
                    onCancel: () => {
                        this.controllerTwo?.close();
                    },
                    onAgree: () => {
                        // 调用napi的底层效果节点
                        const nodeId = util.generateRandomUUID(true);
                        let result = audioNapi.startEnvEffect(this.currentInputId, nodeId, this.selectedEnvironment,
                            this.selectedNode);
                        if (result !== 0) {
                            this.getUIContext().showAlertDialog({
                                message: '添加环境节点失败',
                                autoCancel: true
                            });
                            return;
                        }
                        //添加效果节点到混音台---数据处理
                        let node: Node = {
                            id: nodeId,
                            type: this.selectedEffectNode,
                            color: ColorMap.get(this.selectedEffectNode),
                            environmentType: this.selectedEnvironment
                        }
                        this.isLeft ? this.addEffectNodeToInputList(node) : this.addEffectNodeToMixerList(node);

                        this.controllerTwo?.close();
                    }
                })
            } else if (this.selectedEffectNode === NodeType.NR) {
                NoiseReductionComponent({
                    onCancel: () => {
                        this.controllerTwo?.close();
                    },
                    onAgree: () => {
                        // 调用napi的底层效果节点

                        //添加效果节点到混音台---数据处理
                        const nodeId = util.generateRandomUUID(true);
                        //预留编辑降噪功能方法
                        let result = audioNapi.addNoiseReduction(nodeId, this.currentInputId, this.selectedNode);
                        if (result !== 0) {
                            this.getUIContext().showAlertDialog({
                                message: '添加声场节点失败',
                                autoCancel: true
                            });
                            return;
                        }
                        let node: Node = {
                            id: nodeId,
                            type: this.selectedEffectNode,
                            color: ColorMap.get(this.selectedEffectNode)
                        }
                        this.isLeft ? this.addEffectNodeToInputList(node) : this.addEffectNodeToMixerList(node);

                        this.controllerTwo?.close();
                    }
                })
            } else if (this.selectedEffectNode === NodeType.VB) {
                VoiceBeautifierComponent({
                    selectedVoiceBeautifier: this.selectedVoiceBeautifier,
                    onCancel: () => {
                        this.controllerTwo?.close();
                    },
                    onAgree: () => {
                        // 调用napi的底层效果节点
                        let voiceBeautifierId = util.generateRandomUUID(true);
                        Logger.error(TAG, `-----${this.selectedNode}`)
                        let result =
                            audioNapi.startVBEffect(this.currentInputId, this.selectedVoiceBeautifier,
                                voiceBeautifierId, this.selectedNode);
                        if (result !== 0) {
                            this.getUIContext().showAlertDialog({
                                message: '添加美化节点失败',
                                autoCancel: true
                            });
                            return;
                        }
                        //添加效果节点到混音台---数据处理
                        let node: Node = {
                            id: voiceBeautifierId,
                            type: this.selectedEffectNode,
                            color: ColorMap.get(this.selectedEffectNode),
                            voiceBeautifierType: this.selectedVoiceBeautifier
                        }
                        this.isLeft ? this.addEffectNodeToInputList(node) : this.addEffectNodeToMixerList(node);

                        this.controllerTwo?.close();
                    }
                })
            } else if (this.selectedEffectNode === NodeType.SEPARATION) {
                AudioSeparationComponent({
                    selectedAudioSeparation: this.selectedAudioSeparation,
                    onCancel: () => {
                        this.controllerTwo?.close();
                    },
                    onAgree: () => {
                        // 调用napi的底层效果节点
                        const nodeId = util.generateRandomUUID(true);
                        let result =
                            audioNapi.addAudioSeparation(this.selectedAudioSeparation, nodeId, this.currentInputId,
                                this.selectedNode);
                        if (result !== 0) {
                            this.getUIContext().showAlertDialog({
                                message: `添加${AudioSeparationMap.get(this.selectedAudioSeparation)}失败`,
                                autoCancel: true
                            });
                            return;
                        }
                        //添加效果节点到混音台---数据处理
                        let node: Node = {
                            id: nodeId,
                            type: this.selectedEffectNode,
                            color: ColorMap.get(this.selectedEffectNode),
                            audioSeparationType: this.selectedAudioSeparation
                        }
                        this.isLeft ? this.addEffectNodeToInputList(node) : this.addEffectNodeToMixerList(node);

                        this.controllerTwo?.close();
                    }
                })
            }
        }
    }

    addEffectNodeToMixerList(node: Node) {
        const index = this.mixerList.findIndex(node => node.id === this.selectedNode);
        if (index !== -1) {
            Logger.info(TAG, `addEffectNodeToMixerList mixerList length: ${this.mixerList.length}`);
            this.mixerList.splice(index + 1, 0, node);
        } else {
            this.mixerList.unshift(node);
        }
    }

    addEffectNodeToInputList(node: Node) {
        if (this.inputList.find((item: SongInfo) => item?.nodeId === this.selectedNode)) {
            //如果长按+号的节点是输入节点，则把新添加的效果节点加入到对应的nodes中
            let songInfo = this.nodeList.get(this.selectedNode);
            if (songInfo) {
                let index = (songInfo.nodes ?? []).findIndex(node => node.id === this.selectedNode);
                if (index !== -1) {
                    songInfo.nodes?.splice(index + 1, 0, node);
                }
                this.nodeList.set(this.selectedNode, songInfo);
                Logger.error(TAG, `${this.nodeList.get(songInfo?.nodeId ?? '')?.nodes?.length} --- `);
            }
        } else {
            //如果长按+号的节点是效果节点
            const inputId = this.currentInputId;
            if (inputId) {
                let songInfo = this.nodeList.get(inputId);
                if (songInfo) {
                    // 新增的效果节点插入到长按的按钮后边
                    let index = (songInfo.nodes ?? []).findIndex(node => node.id === this.selectedNode);
                    songInfo.nodes?.splice(index + 1, 0, node);
                    this.nodeList.set(inputId, songInfo);
                    Logger.error(TAG, `${this.nodeList.get(songInfo?.nodeId ?? '')?.nodes?.length} --- `);
                }
            }
        }
    }
}

//"?"的弹窗
@CustomDialog
struct NodeDetailDialog {
    @Prop nodeType: NodeType;
    @Prop isLeft: boolean;
    @Prop selectedNode: string;
    @Prop currentInputId: string;
    @Link mixerList: Array<Node>;
    @Link inputList: SongInfo[];
    @Link nodeList: Map<string, SongInfo>;
    nodeDetailController?: CustomDialogController;
    private balanceBandGains: number[] = [];
    private soundFiledType = '-';
    private environmentType = '-';
    private voiceBeautifierType = '-';
    private audioSeparation = '-';
    //input节点参数
    private songName = '';
    private singerName = '';
    private channels = -1;
    private sampleRate = -1;
    private bitsPerSample = -1;

    aboutToAppear(): void {
        let mixerNode: Node | undefined = this.mixerList.find(node => node.id === this.selectedNode) || undefined;
        let leftNode: SongInfo | undefined =
            this.inputList.find(node => node?.nodeId === this.selectedNode) || undefined;
        if (this.isLeft) {
            //当前是输入节点
            if (leftNode) {
                let inputNode = this.nodeList.get(this.selectedNode);
                if (inputNode) {
                    this.songName = leftNode.songName as string;
                    this.singerName = leftNode.singerName as string;
                    this.channels = inputNode.channels ?? -1;
                    this.sampleRate = inputNode.sampleRate ?? -1;
                    this.bitsPerSample = inputNode.bitsPerSample ?? -1;
                }
            } else {
                //当前是效果节点
                let leftEffectNode: Node | undefined =
                    this.nodeList.get(this.currentInputId)?.nodes?.find(node => node.id === this.selectedNode)
                //在这里添加getOption的逻辑,入参仅为nodeId
                if (leftEffectNode) {
                    let type = audioNapi.getOptions(leftEffectNode.id)
                    Logger.warn(TAG, `type is ${parseInt(type)}---}`)
                    if (leftEffectNode?.type === NodeType.EQ) {
                        this.balanceBandGains = (JSON.parse(type) as number[]) ?? [];
                    } else if (leftEffectNode?.type === NodeType.FIELD) {
                        this.soundFiledType = SoundFiledMap.get(parseInt(type) as SoundFiledType) ?? '-';
                    } else if (leftEffectNode?.type === NodeType.ENV) {
                        this.environmentType = EnvironmentMap.get(parseInt(type) as EnvironmentType) ?? '-';
                    } else if (leftEffectNode?.type === NodeType.VB) {
                        this.voiceBeautifierType = VoiceBeautifierMap.get(parseInt(type) as VoiceBeautifierType) ?? '-';
                    } else if (leftEffectNode?.type === NodeType.SEPARATION) {
                        this.audioSeparation = AudioSeparationMap.get(leftEffectNode.audioSeparationType) ?? '-';
                    }
                }
            }
        } else {
            //右侧初始化赋值
            let type = audioNapi.getOptions(mixerNode?.id)
            if (mixerNode?.type === NodeType.EQ) {
                this.balanceBandGains = (JSON.parse(type) as number[]) ?? [];
            } else if (mixerNode?.type === NodeType.FIELD) {
                this.soundFiledType = SoundFiledMap.get(parseInt(type) as SoundFiledType) ?? '-';
            } else if (mixerNode?.type === NodeType.ENV) {
                this.environmentType = EnvironmentMap.get(parseInt(type) as EnvironmentType) ?? '-';
            } else if (mixerNode?.type === NodeType.VB) {
                this.voiceBeautifierType = VoiceBeautifierMap.get(parseInt(type) as VoiceBeautifierType) ?? '-';
            } else if (mixerNode?.type === NodeType.SEPARATION) {
                this.audioSeparation = AudioSeparationMap.get(mixerNode.audioSeparationType) ?? '-';
            }
        }
    }

    build() {
        Column() {
            Text('节点信息')
                .fontSize(20)
                .fontWeight(FontWeight.Bold)
                .margin({ top: $r('app.float.margin_10'), bottom: $r('app.float.margin_10') })
            Row() {

                  Text(`节点类型：`)
                      .fontWeight(FontWeight.Bold)
                


                      Text(`${this.nodeType}`)
                
            }

            if (this.nodeType === NodeType.EQ) {
                Row() {
                    Text('均衡参数: ')
                        .fontWeight(FontWeight.Bold)
                    if (this.balanceBandGains.length > 0) {
                        ForEach(this.balanceBandGains, (item: number, index) => {

                              Text(`${item}`)
                                  .margin(5)
                            
                        })
                    } else {
                        Text('-')
                    }
                }
            }
            if (this.nodeType === NodeType.FIELD) {
                Row() {

                      Text(`声场参数：`)
                          .fontWeight(FontWeight.Bold)
                    


                      Text(`${this.soundFiledType}`)
                    
                }
            }
            if (this.nodeType === NodeType.ENV) {
                Row() {

                      Text(`环境参数：`)
                          .fontWeight(FontWeight.Bold)
                    


                      Text(`${this.environmentType}`)
                    
                }
            }
            if (this.nodeType === NodeType.VB) {
                Row() {

                      Text(`美化参数：`)
                          .fontWeight(FontWeight.Bold)
                    


                      Text(`${this.voiceBeautifierType}`)
                    
                }
            }
            if (this.nodeType === NodeType.SEPARATION) {
                Row() {

                      Text(`音源分离类型：`)
                          .fontWeight(FontWeight.Bold)
                    


                      Text(`${this.audioSeparation}`)
                    
                }
            }
            if (this.nodeType === NodeType.INPUT) {
                Column() {
                    Text('音频参数：')
                        .fontWeight(FontWeight.Bold)

                      Text(`歌曲名称: ${this.songName}`)
                    


                      Text(`表演者: ${this.singerName ?? '-'}`)
                    


                      Text(`声道: ${this.channels}`)
                    


                      Text(`采样率: ${this.sampleRate}`)
                    


                      Text(`位深: ${this.bitsPerSample}`)
                    
                }.padding({ top: $r('app.float.padding_20') })
            }
            Button('确定')
                .onClick(() => {
                    if (this.nodeDetailController != undefined) {
                        this.nodeDetailController.close();
                    }
                })
                .margin(20)
        }
    }
}

//"*"的弹窗
@CustomDialog
struct SetNodeDialog {
    @Prop isLeft: boolean;
    @Prop selectedNode: string;
    @Prop nodeType: NodeType;
    @Prop currentInputId: string;
    @Link mixerList: Array<Node>;
    @Link inputList: SongInfo[];
    @Link nodeList: Map<string, SongInfo>;
    // 均衡频带
    @State balanceBandGains: number[] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    //声场数据，默认是无
    @State selectedSoundFiled: SoundFiledType = SoundFiledType.SOUND_FIELD_CLOSE;
    //环境数据，默认是无
    @State selectedEnvironment: EnvironmentType = EnvironmentType.ENVIRONMENT_TYPE_CLOSE;
    //美化数据，默认是普通
    @State selectedVoiceBeautifier: VoiceBeautifierType = VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_NORMAL;
    //音源分离
    selectedAudioSeparation: AudioSeparationType = AudioSeparationType.AUDIO_NODE_HUMAN_SOUND_OUTPORT_TYPE;
    setNodeController?: CustomDialogController;

    aboutToAppear(): void {
        let leftNode: SongInfo | undefined =
            this.inputList.find(node => node?.nodeId === this.selectedNode) || undefined
        if (this.isLeft) {
            let inputNode = this.nodeList.get(this.selectedNode);
            //左侧的效果节点修改
            if (!inputNode) {
                let leftEffectNode: Node | undefined =
                    this.nodeList.get(this.currentInputId)?.nodes?.find(node => node.id === this.selectedNode)
                Logger.info(TAG, `node env: ${leftEffectNode?.id}`)
                if (leftEffectNode) {
                    if (leftEffectNode?.balanceBandGains) {
                        this.balanceBandGains = leftEffectNode.balanceBandGains;
                    }
                    if (leftEffectNode?.soundFiledType !== undefined) {
                        this.selectedSoundFiled = leftEffectNode.soundFiledType;
                    }
                    if (leftEffectNode?.environmentType !== undefined) {
                        this.selectedEnvironment = leftEffectNode.environmentType;
                    }
                    if (leftEffectNode?.voiceBeautifierType !== undefined) {
                        this.selectedVoiceBeautifier = leftEffectNode.voiceBeautifierType;
                    }
                    if (leftEffectNode?.audioSeparationType !== undefined) {
                        this.selectedAudioSeparation = leftEffectNode.audioSeparationType
                    }
                }
            }
        } else {
            let node = this.mixerList.find(node => node.id === this.selectedNode);
            Logger.info(TAG, `node env: ${node?.id}`)
            if (node?.balanceBandGains) {
                this.balanceBandGains = node.balanceBandGains;
            }
            if (node?.soundFiledType !== undefined) {
                this.selectedSoundFiled = node.soundFiledType;
            }
            if (node?.environmentType !== undefined) {
                this.selectedEnvironment = node.environmentType;
            }
            if (node?.voiceBeautifierType !== undefined) {
                this.selectedVoiceBeautifier = node.voiceBeautifierType;
            }
            if (node?.audioSeparationType !== undefined) {
                this.selectedAudioSeparation = node.audioSeparationType;
            }
        }
    }

    build() {

          Row() {
              if (this.nodeType === NodeType.EQ) {
                  EquailizerComponent({
                      balanceBandGains: this.balanceBandGains,
                      onCancel: () => {
                          this.setNodeController?.close();
                      },
                      onAgree: () => {
                          //修改效果节点---数据处理
                          const node = this.isLeft ?
                              this.nodeList.get(this.currentInputId)?.nodes?.find(node => node.id ===
                              this.selectedNode) :
                              this.mixerList.find(node => node.id === this.selectedNode)
                          if (node) {
                              let result =
                                  audioNapi.setEqualizerFrequencyBandGains(this.balanceBandGains, this.selectedNode,
                                      this.currentInputId);
                              if (result !== 0) {
                                  Logger.error(TAG, `setEqualizerFrequencyBandGains result: ${result}`);
                                  this.getUIContext().showAlertDialog({
                                      message: '修改均衡节点失败',
                                      autoCancel: true
                                  });
                                  return;
                              }
                              node.balanceBandGains = this.balanceBandGains;
                          } else {
                              this.getUIContext().showAlertDialog({
                                  message: 'not find node in nodeList or mixerList',
                                  autoCancel: true
                              });
                              return;
                          }

                          this.setNodeController?.close();
                      }
                  })
              } else if (this.nodeType === NodeType.FIELD) {
                  FieldComponent({
                      selectedSoundFiled: this.selectedSoundFiled,
                      onCancel: () => {
                          this.setNodeController?.close();
                      },
                      onAgree: () => {
                          //修改效果节点---数据处理
                          const node = this.isLeft ?
                              this.nodeList.get(this.currentInputId)?.nodes?.find(node => node.id ===
                              this.selectedNode) :
                              this.mixerList.find(node => node.id === this.selectedNode)
                          if (node) {
                              let result = audioNapi.resetFieldEffect(this.currentInputId, this.selectedSoundFiled,
                                  this.selectedNode);
                              if (result !== 0) {
                                  this.getUIContext().showAlertDialog({
                                      message: '设置声场节点失败',
                                      autoCancel: true
                                  });
                                  return;
                              }
                              node.soundFiledType = this.selectedSoundFiled;
                          } else {
                              this.getUIContext().showAlertDialog({
                                  message: 'not find node in nodeList or mixerList',
                                  autoCancel: true
                              });
                              return;
                          }

                          this.setNodeController?.close();
                      }
                  })
              } else if (this.nodeType === NodeType.ENV) {
                  EnvironmentComponent({
                      selectedEnvironment: this.selectedEnvironment,
                      onCancel: () => {
                          this.setNodeController?.close();
                      },
                      onAgree: () => {
                          //修改效果节点---数据处理
                          const node = this.isLeft ?
                              this.nodeList.get(this.currentInputId)?.nodes?.find(node => node.id ===
                              this.selectedNode) :
                              this.mixerList.find(node => node.id === this.selectedNode)
                          if (node) {
                              let result = audioNapi.resetEnvEffect(this.currentInputId, this.selectedNode,
                                  this.selectedEnvironment);
                              if (result !== 0) {
                                  this.getUIContext().showAlertDialog({
                                      message: '设置环境节点失败',
                                      autoCancel: true
                                  });
                                  return;
                              }
                              node.environmentType = this.selectedEnvironment;
                          } else {
                              this.getUIContext().showAlertDialog({
                                  message: 'not find node in nodeList or mixerList',
                                  autoCancel: true
                              });
                              return;
                          }

                          this.setNodeController?.close();
                      }
                  })
              } else if (this.nodeType === NodeType.VB) {
                  VoiceBeautifierComponent({
                      selectedVoiceBeautifier: this.selectedVoiceBeautifier,
                      onCancel: () => {
                          this.setNodeController?.close();
                      },
                      onAgree: () => {
                          // 调用napi的底层效果节点
                          const node = this.isLeft ?
                              this.nodeList.get(this.currentInputId)?.nodes?.find(node => node.id ===
                              this.selectedNode) :
                              this.mixerList.find(node => node.id === this.selectedNode)
                          if (node) {
                              let result =
                                  audioNapi.resetVBEffect(this.currentInputId, this.selectedVoiceBeautifier,
                                      this.selectedNode);
                              if (result !== 0) {
                                  this.getUIContext().showAlertDialog({
                                      message: '添加美化节点失败',
                                      autoCancel: true
                                  });
                                  return;
                              }
                          }

                          //修改效果节点---数据处理
                          if (this.isLeft) {
                              let inputNode = this.nodeList.get(this.selectedNode);
                              //左侧的效果节点修改
                              if (!inputNode) {
                                  let leftEffectNode: Node | undefined =
                                      this.nodeList.get(this.currentInputId)?.nodes?.find(node => node.id ===
                                      this.selectedNode)
                                  Logger.error(TAG, `node env: ${leftEffectNode?.environmentType}`)
                                  if (leftEffectNode) {
                                      leftEffectNode.voiceBeautifierType = this.selectedVoiceBeautifier;
                                  }
                              }
                          } else {
                              const node = this.mixerList.find(node => node.id === this.selectedNode);
                              if (node?.voiceBeautifierType !== undefined) {
                                  node.voiceBeautifierType = this.selectedVoiceBeautifier;
                              }
                          }

                          this.setNodeController?.close();
                      }
                  })
              }
          }
        
    }
}

@CustomDialog
@Component
export struct CheckNodeDialog {
    //传入的数据
    //区分混音台左边区域和右边区域
    @Prop isLeft: boolean;
    //当前选中的节点类型
    @Prop nodeType: NodeType;
    //当前选中的节点ID
    @Prop selectedNode: string;
    //混音台右边数据列表
    @Link mixerList: Array<Node>;
    //混音台左边输入节点列表
    @Link inputList: SongInfo[];
    //混音台左边节点列表
    @Link nodeList: Map<string, SongInfo>;
    @Link isPlayRealTime: boolean;
    //创建管线，工作模式
    @Link param: number;
    //当前音轨的inputId
    @State currentInputId: string = '';
    //组件本身数据
    private nodes: Array<NodeType> = [];
    private select: number | number[] = 0;
    @State selectedEffectNode: NodeType = NodeType.EQ;

    aboutToAppear(): void {
        this.currentInputId = this.findInputId();
        //获取当前设配支持的效果节点列表
        let nodeType: number[] = audioNapi.getEffectNodeList();
        this.nodes = nodeType
            .map(num => NumberToNodeTypeMap.get(num)) // 将数字转换为 NodeType
            .filter(type => type !== undefined) as NodeType[]; // 过滤掉 undefined 值
    }

    effectNodeDialog: CustomDialogController | null = new CustomDialogController({
        builder: EffectNodeDialog({
            selectedNode: this.selectedNode,
            selectedEffectNode: this.selectedEffectNode,
            mixerList: this.mixerList,
            currentInputId: this.currentInputId,
            isLeft: this.isLeft,
            inputList: this.inputList,
            nodeList: this.nodeList
        }),
    })
    NodeDetailDialogController: CustomDialogController | null = new CustomDialogController({
        builder: NodeDetailDialog({
            mixerList: this.mixerList,
            isLeft: this.isLeft,
            selectedNode: this.selectedNode,
            inputList: this.inputList,
            nodeList: this.nodeList,
            nodeType: this.nodeType,
            currentInputId: this.currentInputId
        })
    })
    SetNodeDialogController: CustomDialogController | null = new CustomDialogController({
        builder: SetNodeDialog({
            mixerList: this.mixerList,
            isLeft: this.isLeft,
            selectedNode: this.selectedNode,
            inputList: this.inputList,
            nodeList: this.nodeList,
            nodeType: this.nodeType,
            currentInputId: this.currentInputId
        })
    })
    controller?: CustomDialogController;
    // 若尝试在CustomDialog中传入多个其他的Controller，以实现在CustomDialog中打开另一个或另一些CustomDialog，那么此处需要将指向自己的controller放在所有controller的后面
    cancel: () => void = () => {
    }
    confirm: () => void = () => {
    }

    private getIconResource(icon: string): Resource {
        const map: Record<string, Resource> = {
            '+': $r('app.media.plus'),
            '-': $r('app.media.minus_circle'),
            '?': $r('app.media.questionmark_circle'),
            '*': $r('app.media.star')
        };
        return map[icon];
    }

    findInputId(): string {
        for (const entry of this.nodeList.entries()) {
            const key = entry[0];
            const songInfo = entry[1];

            let selectedNode: string = this.selectedNode;
            const foundNode = songInfo.nodes?.find((node) => node.id === selectedNode);
            if (foundNode) {
                return key;
            } else if (this.inputList.find((node) => node?.nodeId === selectedNode)) {
                Logger.error(TAG, `this selected is input node : ${selectedNode}, inputId is ${selectedNode}`)
                return selectedNode;
            }
        }
        return '';
    }

    //"+"的弹窗
    selectNodeHandle() {
        // 根据 param 过滤可选节点类型
        let filteredNodes: NodeType[] = this.nodes;
        if (this.param === 2) {
            // 仅显示“均衡”
            filteredNodes = [NodeType.EQ];
            this.getUIContext().showAlertDialog({
                title: '添加节点',
                message: '当前模式仅支持添加“均衡”节点',
                autoCancel: false, // 不允许点击外部关闭
                primaryButton: {
                    value: '取消',
                    action: () => {
                        Logger.info(TAG, '取消添加均衡节点');
                    }
                },
                secondaryButton: {
                    value: '确定',
                    style: DialogButtonStyle.HIGHLIGHT,
                    defaultFocus: true,
                    action: () => {
                        Logger.info(TAG, '确认添加均衡节点');
                        // 直接设置选中为 EQ，打开效果弹窗
                        this.selectedEffectNode = NodeType.EQ;
                        this.select = this.nodes.indexOf(NodeType.EQ); // 保持选中状态一致
                        this.effectNodeDialog?.open();
                    }
                }
            });
        } else {
            // 默认：全部显示（兼容旧逻辑）
            filteredNodes = this.nodes;
            this.getUIContext().showTextPickerDialog({
                range: filteredNodes,
                selected: this.select,
                value: this.selectedEffectNode,
                defaultPickerItemHeight: 40,
                onAccept: (value: TextPickerResult) => {
                    Logger.info(TAG, `selectNodeHandle onAccept, value: ${JSON.stringify(value)}`);
                    // 设置select为按下确定按钮时候的选中项index，这样当弹窗再次弹出时显示选中的是上一次确定的选项
                    this.select = value.index;
                    // 点击确定后，被选到的文本数据展示到页面
                    this.selectedEffectNode = value.value as NodeType;

                    this.effectNodeDialog?.open();
                },
                onCancel: () => {
                    Logger.info(TAG, 'selectNodeHandle onCancel');
                },
                onChange: () => {
                    Logger.info(TAG, 'selectNodeHandle onChange');
                }
            })
        }
    }

    //"-"的弹窗
    deleteNodeHandle() {
        this.getUIContext().showAlertDialog(
            {
                title: '确认删除该节点',
                message: '',
                autoCancel: true,
                alignment: DialogAlignment.Bottom,
                gridCount: 4,
                offset: { dx: 0, dy: -20 },
                primaryButton: {
                    value: '取消',
                    action: () => {
                        console.info('Callback when the first button is clicked');
                    }
                },
                secondaryButton: {
                    enabled: true,
                    defaultFocus: true,
                    style: DialogButtonStyle.HIGHLIGHT,
                    value: '确定',
                    action: () => {
                        if (this.isLeft) {
                            if (this.inputList.find((item: SongInfo) => item?.nodeId === this.selectedNode)) {
                                let result = audioNapi.deleteSong(this.selectedNode);
                                if (result !== 0) {
                                    this.getUIContext().showAlertDialog({
                                        message: '删除节点失败',
                                        autoCancel: true
                                    });
                                    return;
                                }
                                //如果点击输入节点的-号，则把inputList和nodeList都要处理
                                this.nodeList.delete(this.selectedNode);
                                const index = this.inputList.findIndex(item => item?.nodeId === this.selectedNode);
                                if (index !== -1) {
                                    this.inputList[index] = undefined;
                                }
                            } else {
                                const inputId = this.findInputId();
                                let songInfo = this.nodeList.get(inputId);
                                if (songInfo) {
                                    // 删除对应节点
                                    let index = songInfo.nodes?.findIndex(node => node.id === this.selectedNode);
                                    if (index !== -1) {
                                        let result = audioNapi.deleteNode(this.selectedNode);
                                        if (result !== 0) {
                                            this.getUIContext().showAlertDialog({
                                                message: '删除节点失败',
                                                autoCancel: true
                                            });
                                            return;
                                        }
                                        songInfo.nodes?.splice(index, 1);
                                        this.nodeList.set(inputId, songInfo);
                                    } else {
                                        this.getUIContext().showAlertDialog({
                                            message: 'deleteNodeHandle nodeList is not selectedNodeId',
                                            autoCancel: true
                                        });
                                    }

                                    this.controller?.close();
                                }
                            }
                        } else {
                            Logger.info(TAG, `deleteNodeHandle selectedNode: ${this.selectedNode}`);
                            const index = this.mixerList.findIndex(node => node.id === this.selectedNode);
                            if (index !== -1) {
                                let result = audioNapi.deleteNode(this.selectedNode);
                                if (result !== 0) {
                                    this.getUIContext().showAlertDialog({
                                        message: '删除节点失败',
                                        autoCancel: true
                                    });
                                    return;
                                }
                                this.mixerList.splice(index, 1);
                                this.controller?.close();
                            } else {
                                this.getUIContext().showAlertDialog({
                                    message: 'deleteNodeHandle mixerList is not has selectedNodeId',
                                    autoCancel: true
                                });
                            }
                        }
                    }
                }
            }
        )
    }

    build() {
        Column() {
            Text('').fontSize(20).margin({ top: 10, bottom: 10 })
            Flex({ justifyContent: FlexAlign.SpaceAround }) {
                if (this.nodeType !== NodeType.OUTPUT && !this.isPlayRealTime &&
                    this.nodeType !== NodeType.SEPARATION) {
                    Image(this.getIconResource('+'))
                        .width('50')
                        .height('50')
                        .onClick(() => {
                            this.selectNodeHandle();
                        })
                }
                if (this.nodeType !== NodeType.OUTPUT && this.nodeType !== NodeType.MIXER) {
                    Image(this.getIconResource('-'))
                        .width('50')
                        .height('50')
                        .onClick(() => {
                            this.deleteNodeHandle();
                            this.controller?.close();
                        })
                }
                Image(this.getIconResource('?'))
                    .width('50')
                    .height('50')
                    .onClick(() => {
                        if (this.NodeDetailDialogController !== null) {
                            this.NodeDetailDialogController.open();
                        }
                    })
                if (this.nodeType !== NodeType.INPUT && this.nodeType !== NodeType.MIXER &&
                    this.nodeType !== NodeType.OUTPUT &&
                    this.nodeType !== NodeType.NR) {
                    Image(this.getIconResource('*'))
                        .width('50')
                        .height('50')
                        .onClick(() => {
                            if (this.SetNodeDialogController !== null) {
                                this.SetNodeDialogController.open();
                            }
                        })
                }
            }.margin({ bottom: 10 })
        }.borderRadius(10)
        // 如果需要使用border属性或cornerRadius属性，请和borderRadius属性一起使用。
    }
}