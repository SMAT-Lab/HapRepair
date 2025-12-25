
/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. All rights reserved.
 */
import { StyleConst } from "../utils/StyleConst";
import { Node, NodeType, ColorMap, SongInfo } from "../utils/importSongs/InterfaceInfo";
import { Logger } from '../utils/Logger';
import audioNapi from 'libentry.so';
import { importSingleSong } from '../utils/importSongs/ImportSongs';
import { SetFormatDialog } from './customDialog/SetFormatDialog';
import { CheckNodeDialog } from './customDialog/CheckNodeDialog';
import { JSON, util } from '@kit.ArkTS';
import { saveFileBuffer, RealTimeSaveFileBuffer, MainTitleTextModifier } from '../utils/ExportFile';
import { taskpool } from '@kit.ArkTS';
import { fileUri, picker } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';
import fs, { ReadOptions } from '@ohos.file.fs';
import { AlertDialog, SymbolGlyphModifier, window } from '@kit.ArkUI';
import display from '@ohos.display';

import { getDateStringWithTimeStamp } from '../utils/Util';

const TAG: string = 'AudioEditTestApp_MixingConsole';

@Builder
export function MixingConsoleBuilder() {
    MixingConsole()
}

@Component
export struct MixingConsole {
    pageInfos: NavPathStack = new NavPathStack();
    // 主标题样式修改器
    @State mainTitleModifier: MainTitleTextModifier = new MainTitleTextModifier();
    private symbolModifier: SymbolGlyphModifier =
        new SymbolGlyphModifier($r('sys.symbol.chevron_backward')).fontColor([$r('app.color.index_title_text_color')]);
    // 判断是否是横屏
    @StorageLink('isHorizontalScreen') isHorizontalScreen: boolean = false;

    @State param: number = 0;
    @State inputList: SongInfo[] = [];
    private MAX_TRACKS: number = 5; // 最大音轨数
    @State nodeList: Map<string, SongInfo> = new Map();
    private wavBuffer: ArrayBuffer = new ArrayBuffer(0);
    private fd: number = -1;
    // 是否录制
    @State g_isRecord: boolean = false;
    // 导出中
    isExported: boolean = false;
    @State mixerList: Array<Node> = [];
    @State mixerNode: Node = {
        id: util.generateRandomUUID(true),
        type: NodeType.MIXER,
        color: ColorMap[NodeType.MIXER]
    }
    @State outputNode: Node = {
        id: util.generateRandomUUID(true),
        type: NodeType.OUTPUT,
        color: ColorMap[NodeType.OUTPUT]
    }
    // 是否播放
    @State isPlay: boolean = false;
    @State isPlayRealTime: boolean = false;
    // 导出样式
    @State newAudioFileName: string = '';
    audioFormate: string = '.wav';
    @State sampleRate: number = -1;
    @State channels: number = -1;
    @State bitsPerSample: number = -1;
    // 节点操作弹窗参数
    @State isLeft: boolean = true;
    @State selectedNode: string = '';
    @State nodeType: NodeType = NodeType.INPUT;
    finished: boolean = false;

    aboutToAppear(): void {
        Logger.info(TAG, 'aboutToAppear');
        let context = this.getUIContext()?.getHostContext() as common.UIAbilityContext;
        window.getLastWindow(context).then((windowClass) => {
            // 设置屏幕横屏
            windowClass.setPreferredOrientation(window.Orientation.AUTO_ROTATION_LANDSCAPE);
        });
        this.isPlayRealTime = false;
        // 初始化空列表（保留5个空位）
        this.inputList = Array(this.MAX_TRACKS).fill(null);
        
        audioNapi.registerFinishedCallback((result: boolean) => {
            this.finished = result;
            Logger.info(TAG, `registerFinishedCallback finished: ${JSON.stringify(this.finished)}`);
            if (this.finished) {
                if (this.g_isRecord) {
                    taskpool.execute(RealTimeSaveFileBuffer).then(async (value: object) => {
                        Logger.info(TAG, 'taskpool onClick then');
                        this.SaveBuffer(value as ArrayBuffer);
                    });
                }
                this.isExported = false;
                this.g_isRecord = false;
                this.isPlay = false;
            }
        })

        audioNapi.registerAudioFormatCallback((result: string[]) => {
            Logger.info(TAG, `registerAudioFormatCallback string size: ${JSON.stringify(result.length)}, result: ${result}`);
            this.sampleRate = Number(result[0]);
            this.channels = Number(result[1]);
            switch (Number(result[2])) {
                case 0:
                    this.bitsPerSample = 8;
                    break;
                case 1:
                    this.bitsPerSample = 16;
                    break;
                case 2:
                    this.bitsPerSample = 24;
                    break;
                case 3:
                    this.bitsPerSample = 32;
                    break;
                case 4:
                    this.bitsPerSample = 32;
                    break;
            }
        })
    }

    aboutToDisappear(): void {
        Logger.info(TAG, 'aboutToDisappear');
        // 销毁播放器
        try {
            let result = audioNapi.audioRendererDestory();
            Logger.info(TAG, `audioRendererDestory result: ${result}`);
        } catch (e) {
            Logger.info(TAG, `audioRendererDestory error: ${JSON.stringify(e)}`);
        }
        // 销毁管线， 销毁引擎
        try {
            let result = audioNapi.audioEditDestory();
            Logger.info(TAG, `audioEditDestory result: ${result}`);
        } catch (e) {
            Logger.info(TAG, `audioEditDestory error: ${JSON.stringify(e)}`);
        }
        let context = this.getUIContext()?.getHostContext() as common.UIAbilityContext;
        window.getLastWindow(context).then((windowClass) => {
            windowClass.setPreferredOrientation(window.Orientation.AUTO_ROTATION);
        });
    }

    // 设置弹窗
    dialogControllerSet: CustomDialogController | null = new CustomDialogController({
        builder: SetFormatDialog({
            newAudioFileName: this.newAudioFileName,
            sampleRate: this.sampleRate,
            channels: this.channels,
            bitsPerSample: this.bitsPerSample,
            nodeList: this.nodeList,
            cancel: () => {

            },
            confirm: () => {

            }
        }),
        maskColor: $r('app.color.mixing_console_mask_color'),
    });

    // 查看效果节点信息弹窗
    checkNodeDialogController: CustomDialogController | null = new CustomDialogController({
        builder: CheckNodeDialog({
            isLeft: this.isLeft,
            selectedNode: this.selectedNode,
            mixerList: this.mixerList,
            inputList: this.inputList,
            nodeList: this.nodeList,
            nodeType: this.nodeType,
            isPlayRealTime: this.isPlayRealTime,
            param: this.param,
            cancel: () => {

            },
            confirm: () => {

            }
        }),
    });

    // 添加output节点弹窗
    addOutputDialogController: CustomDialogController = new CustomDialogController({
        builder: AlertDialog({
            primaryTitle: '添加OUTPUT节点',
            primaryButton: {
                value: '取消'
            },
            secondaryButton: {
                value: '确认',
                role: ButtonRole.ERROR,
                action: () => {
                    const outputId = util.generateRandomUUID(true);
                    this.outputNode.id = outputId;
                }
            }
        }),
    });

    build() {
        NavDestination() {
            Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
                Row() {
                    Image($r('app.media.ic_set'))
                        .width('30')
                        .height('30')
                        .onClick(() => {
                            // 弹窗，导出的音频格式
                            this.dialogControllerSet?.open();
                        })

                }
                Row() {
                    // 保存
                    Image($r('app.media.ic_save'))
                        .width('30')
                        .height('30')
                        .onClick(() => {
                            taskpool.execute(saveFileBuffer).then(async (value: object) => {
                                Logger.info(TAG, 'taskpool onClick save mixingConsole start');
                                this.SaveBuffer(value as ArrayBuffer);
                                // 将buffer的map中的 totalWriteAudioDataSize 置为 0
                                audioNapi.resetTotalWriteAudioDataSize();
                            });
                        })
                }.visibility(this.param === 1 ? Visibility.Visible : Visibility.None)
                Row() {
                    // 录制
                    Image(this.g_isRecord ? $r('app.media.ic_record_circle_2') : $r('app.media.ic_record_circle'))
                        .width('30')
                        .height('30')
                        .margin({right: $r('app.float.margin_5')})
                        .onClick(() => {
                            audioNapi.record();
                            this.g_isRecord = true;
                        })
                    // 播放
                    Image(!this.isPlay ? $r('app.media.ic_record_playing') : $r('app.media.ic_record_paused'))
                        .width('30')
                        .height('30')
                        .margin({right: $r('app.float.margin_5')})
                        .onClick(() => {
                            if (this.nodeList.size === 0) {
                                this.getUIContext().showAlertDialog({
                                    message: '至少选中一个音频才可以播放',
                                    confirm: {
                                        value: '确定',
                                        action: () => {
                                        }
                                    },
                                    cancel: () => {
                                        Logger.info(TAG, `cancel`);
                                    }
                                });
                                return;
                            }
                            this.isPlayRealTime = true;
                            this.isPlay = !this.isPlay;
                            // 播放
                            if (this.isPlay) {
                                // 初始化 Renderer
                                audioNapi.audioRendererInit();
                                // 点击播放，才开始 AudioRenderer
                                audioNapi.audioRendererStart();
                            } else {
                                // 暂停
                                audioNapi.audioRendererStop();
                                // 将buffer的map中的 totalWriteAudioDataSize 置为 0
                                audioNapi.resetTotalWriteAudioDataSize();
                            }
                        })
                }
                .alignItems(VerticalAlign.Center)
                .visibility(this.param === 2 ? Visibility.Visible : Visibility.None)
            }
            .height(this.isHorizontalScreen ? '15%' : "10%")
            .width(StyleConst.NINETY_FIVE_PERCENT_WIDTH)
            .borderRadius('10')
            .margin({bottom: $r('app.float.margin_10')})
            .padding({ left: $r('app.float.margin_35'), right: $r('app.float.margin_35') })
            .backgroundColor($r('app.color.index_audio_clip_card'))

            Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween }) {
                // 左侧
                Column() {
                    ForEach(this.inputList, (info: SongInfo, index: number) => {
                        Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.Start }) {
                            Row() {
                                if(info && this.nodeList.has(info.nodeId || '')) {
                                    ForEach(Array.from(this.nodeList.get(info?.nodeId || '')?.nodes || []), (node: Node) => {
                                        Column() {
                                            this.EffectNode(true, node.type, node.id, info)
                                        }.margin({top: $r('app.float.margin_5')})
                                    }, (item, index) => item)
                                } else {
                                    Column() {
                                        this.Node(true, info);
                                    }.onClick((event: ClickEvent) => {
                                        const inputId: string = util.generateRandomUUID(true);
                                        Logger.info(TAG, `inputId: ${inputId}`);
                                        importSingleSong(this.nodeList, inputId).then((songMap: Map<string, SongInfo>) => {
                                            // 将input节点也要加入到nodes里边
                                            let inputInfo: Node = {
                                                id: inputId,
                                                type: NodeType.INPUT,
                                                color: ColorMap.get(NodeType.INPUT)
                                            }
                                            let nodesArray: Array<Node> = [];
                                            nodesArray.push(inputInfo);
                                            let tmpInfo: SongInfo = songMap.entries().next().value;
                                            tmpInfo.nodes = nodesArray;
                                            // 将导入的音频文件存在nodeList中
                                            for (let value of songMap.values()) {
                                                Logger.error(TAG, `${value.nodeId} --- ${inputId}`);
                                                if (value.nodeId === inputId) {
                                                    this.inputList[index] = value;
                                                    this.getPcmFileBuffer(this.inputList[index]);
                                                    if (this.wavBuffer.byteLength !== 0) {
                                                        audioNapi.audioInAndOutInit(inputId, this.outputNode.id, this.mixerNode.id, this.fd, this.wavBuffer.byteLength);
                                                    }
                                                    Logger.info(TAG, `fd: ${this.fd}`);
                                                    fs.closeSync(this.fd);
                                                    let sampleRate = this.sampleRate;
                                                    let channels = this.channels;
                                                    let bitsPerSample = this.bitsPerSample;
                                                    setTimeout(() => {
                                                        if (tmpInfo) {
                                                            tmpInfo.sampleRate = sampleRate;
                                                            tmpInfo.channels = channels;
                                                            tmpInfo.bitsPerSample = bitsPerSample;
                                                        }
                                                    }, 100);
                                                    this.nodeList.set(inputId, tmpInfo);
                                                }
                                            }

                                            Logger.error(TAG, `Node list size: ${this.nodeList.size}`);
                                            Logger.error(TAG, `input list size: ${this.inputList.length}`);
                                            // 超过一个输入就增加混音节点
                                            if (this.nodeList.size > 1 && this.mixerNode.id === '') {
                                                Logger.info(TAG, 'LEFT ADD');
                                                const mixerNodeId = util.generateRandomUUID(true);
                                                this.mixerNode.id = mixerNodeId;
                                            }
                                        });
                                    })
                                    .backgroundColor($r('app.color.index_audio_clip_card'))
                                    .margin({top: $r('app.float.margin_5')})
                                }
                            }
                        }
                        .height('20%')
                        .width('100%')
                    }, (item, index) => item)
                }
                .height('100%')
                .layoutWeight(1)
                .backgroundColor($r('app.color.mixing_console_backgroundColor'))
                .padding({left: $r('app.float.margin_10'), right: $r('app.float.margin_10')})

                Blank().width(1).backgroundColor($r('app.color.index_audio_clip_card'))

                // 右侧
                Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
                    Row() {
                        if (this.nodeList.size > 1) {
                            Column() {
                                this.EffectNode(false, NodeType.MIXER, this.mixerNode.id);
                            }
                        }
                        ForEach(this.mixerList, (node: Node) => {
                            Column() {
                                this.EffectNode(false, node.type, node.id);
                            }
                        }, (item, index) => item)
                        if (this.nodeList.size > 0) {
                            Column() {
                                this.EffectNode(false, NodeType.OUTPUT, this.outputNode.id);
                            }
                        }

                        if (this.outputNode.id === '') {
                            Column() {
                                this.Node(false);
                            }.onClick(() => {
                                if (this.nodeList.size > 0) {
                                    this.addOutputDialogController.open();
                                } else {
                                    this.getUIContext().showAlertDialog({
                                        title: '提示',
                                        message: '添加输入节点后才可以添加输出节点'
                                    });
                                }
                            })
                        }
                    }
                }
                .height('100%')
                .width('38%')
                .backgroundColor($r('app.color.mixing_console_backgroundColor'))
            }
            .height('90%')
            .width(StyleConst.NINETY_FIVE_PERCENT_WIDTH)
            .backgroundColor($r('app.color.index_audio_clip_card'))
        }
        .onReady((context: NavDestinationContext) => {
            this.pageInfos = context.pathStack;
            this.param = context.pathInfo.param as number;
            Logger.info(TAG, `this param is ${this.param}`);
            // 工作模式
            if (this.param !== 0) {
                try {
                    let result = audioNapi.audioEditNodeInit(this.param);
                    Logger.info(TAG, `audioEditNodeInit result: ${result}`);
                } catch (e) {
                    Logger.error(TAG, `audioEditNodeInit error: ${JSON.stringify(e)}`);
                }
            } else {
                // 弹出小弹窗
               Logger.info(TAG, `this param is invalid: ${this.param}`); 
            }
        })
        .title(this.param === 1 ? $r('app.string.mixing_console_cascade') : $r('app.string.mixing_console_playing'),
            {mainTitleModifier: this.mainTitleModifier})
        .height('100%')
        .width('100%')
        .backgroundColor($r('app.color.main_backgroundColor'))
        .backButtonIcon(this.symbolModifier)
    }

    @Builder 
    EffectNode(isLeft : boolean, type : NodeType, nodeId : string, info ?: SongInfo) {
        Column() {
            this.Node(isLeft, info, type);
        }.onClick(() => {
            this.isLeft = isLeft;
            this.selectedNode = nodeId;
            this.nodeType = type;
            Logger.info(TAG, `EffectNode nodeId: ${nodeId} ${isLeft}`);
            if (type !== NodeType.INPUT) {
                this.checkNodeDialogController?.open();
            } else {
                //提前copy一份nodes数据
                let nodesArray = this.nodeList.get(nodeId)?.nodes ?? [];
                importSingleSong(this.nodeList, nodeId).then((songMap: Map<string, SongInfo>) => {
                    Logger.error(TAG, `---- nodesArray length is ${nodesArray.length}`);
                    let songInfo = songMap.get(nodeId)
                    if(songInfo){
                        let inputInfo: Node = {
                            id: nodeId,
                            type: NodeType.INPUT,
                            color: ColorMap.get(NodeType.INPUT)
                        }
                        if(nodesArray.length > 0){
                            nodesArray[0] = inputInfo
                        } else {
                            nodesArray = [inputInfo]
                        }
                        Logger.error(TAG, `nodesArray length is ${nodesArray.length}`);
                        songInfo.nodes = nodesArray;
                        //获取音频的相关参数
                        this.getPcmFileBuffer(songInfo);
                        if(this.wavBuffer.byteLength !== 0) {
                            audioNapi.audioInAndOutInit(nodeId, this.outputNode.id, this.mixerNode.id, this.fd, this.wavBuffer.byteLength);
                            Logger.info(TAG, `fd: %{public}`);
                            setTimeout(() => {
                                if (songInfo) {
                                    songInfo.sampleRate = this.sampleRate;
                                    songInfo.channels = this.channels;
                                    songInfo.bitsPerSample = this.bitsPerSample;
                                }
                            }, 100);
                            fs.closeSync(this.fd);
                        }
                        this.nodeList.set(nodeId, songInfo)
                        const index = this.inputList.findIndex(song => song?.nodeId === nodeId);
                        if(index !== -1 && songInfo.uri !== undefined){
                            this.inputList[index] = songInfo
                        }
                    }
                });
                Logger.error(TAG,`nodeList size is ${this.nodeList.size}`)
            }
        })
        .gesture(
            LongPressGesture()
                .onAction((event: GestureEvent) => {
                    this.isLeft = isLeft;
                    this.selectedNode = nodeId;
                    this.nodeType = type;
                    this.checkNodeDialogController?.open();
                })
        )
    }

    @Builder
    Node(isLeft: boolean, info?:SongInfo | null, type?:NodeType) {
        if ( info && info.songName && type === NodeType.INPUT){

              Text(info.songName)
                  .height('60%')
                  .width("15%")
                  .margin({right: $r('app.float.margin_5')})
                  .textAlign(TextAlign.Center)
                  .backgroundColor(ColorMap.get(type))
                  .fontSize($r('app.float.font_size_10'))
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
                  .borderRadius(5)
            
        } else if (type) {
            Text(type)
                .height(isLeft ? '60%' : '20%')
                .width("10%")
                .margin({right: $r('app.float.margin_5')})
                .textAlign(TextAlign.Center)
                .backgroundColor(ColorMap.get(type))
                .borderRadius(5)
        }   else{
            Text("+")
                .height(isLeft ? '80%' : '20%')
                .width("10%")
                .margin({right: $r('app.float.margin_5')})
                .textAlign(TextAlign.Center)
                .fontColor(Color.White)
        }
    }

    getPcmFileBuffer(info: SongInfo): boolean{
        if(info.songType === 'pcm'){
            return false;
        }
        try {
            let path: string = new fileUri.FileUri(info.uri).path;
            Logger.info(TAG, `getPcmFileBuffer path: ${JSON.stringify(path)}`);
            let file  = fs.openSync(path, fs.OpenMode.READ_ONLY | fs.OpenMode.CREATE);
            let fsStat  = fs.statSync(path);
            Logger.info(TAG, `failSize : ${fsStat.size}`);
            // 不要删 --- let buffer = audioNapi.getFileBuffer(file.fd, fsStat.size);
            let buffer = new ArrayBuffer(fsStat.size);
            let readOption: ReadOptions = {
                offset: 0, // 期望读取文件的位置。可选，默认从当前位置开始读
                length: fsStat.size // 每次期望读取数据的长度。可选，默认缓冲区长度
            }
            fs.readSync(file.fd, buffer, readOption);
            Logger.info(TAG, `getPcmFileBuffer buffer length: ${buffer.byteLength}`);
            const wavView = new Uint8Array(buffer);
            let dataBuffer = wavView.slice(44).buffer;
            Logger.info(TAG, `wavBuffer length: ${buffer.byteLength}`);
            this.wavBuffer = buffer;
            this.fd = file.fd;
            return true;
        } catch (e) {
            Logger.error(TAG, `getPcmFileBuffer error: ${JSON.stringify(e)}`);
            return false;
        }
    }

    async SaveBuffer(value: ArrayBuffer) {
        try {
            let pcmBuffer = value;
            Logger.info(TAG, `SaveBuffer pcmBuffer length : ${pcmBuffer.byteLength}`);
            if(pcmBuffer.byteLength === 0) {
                Logger.info(TAG, 'SaveBuffer error');
            }
            // picker选择器，保存PCM文件
            let documentSaveOptions = new picker.DocumentSaveOptions();
            if(this.newAudioFileName ===  ''){
                this.newAudioFileName = getDateStringWithTimeStamp(new Date().getTime());
            } else {
                this.newAudioFileName = this.newAudioFileName + ' ' + getDateStringWithTimeStamp(new Date().getTime());   
            }
            let newFileNames = `${this.newAudioFileName}${this.audioFormate}`;
            Logger.info(TAG, `SaveBuffer newFileNames is: ${newFileNames}`);
            documentSaveOptions.newFileNames = [newFileNames];
            let context = getContext() as common.Context; // 请确保 getContext(this) 返回结果为UIAbilityContext
            let documentPicker = new picker.DocumentViewPicker(context);
            documentPicker.save(documentSaveOptions, (err: BusinessError, documentSaveResult: Array<string>) => {
                if(err){
                    Logger.error(TAG,
                        `DocumentViewPicker.save failed with err, code is: ${err.code}, message is: ${err.message}`);
                    return;
                }
                Logger.info(TAG, 
                    'DocumentViewPicker.save successfully, documentSaveResult uri: ' + JSON.stringify(documentSaveResult));
                let filePath = documentSaveResult;
                // cdocumentSaveResult数组中只有一个路径
                let file = fs.openSync(documentSaveResult[0], fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
                Logger.info(TAG, `SaveBuffer filePath : ${filePath}`);
                if(this.audioFormate === '.wav') {
                    // 处理wav
                    let wavBuffer = this.writeWavFileHeader(pcmBuffer);
                    let writeLen = fs.writeSync(file.fd, wavBuffer);
                    Logger.info(TAG, `SaveWavBuffer writeLen : ${writeLen}`);
                    fs.closeSync(file);
                }
            }) 
        } catch(e) {
            Logger.error(TAG, `SaveBuffer catch: ${JSON.stringify(e)}`)
        }
    }

    writeString(dv: DataView, offset: number, str: string) {
        for (let i = 0; i < str.length; i++){
            dv.setUint8(offset+i, str.charCodeAt(i));
        }
    }

    concatArrayBuffer(wavHeaderBuffer: ArrayBuffer, pcmBuffer: ArrayBuffer) {
        const viewWavHeader = new Uint8Array(wavHeaderBuffer);
        const viewPcm = new Uint8Array(pcmBuffer);

        // 创建一个新的 ArrayBuffer， 大小为两个原始缓冲区之和
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
            `writeWavFileHeader sampleRate: ${this.sampleRate} --- channels: ${this.channels} --- bitsPerSample: ${this.bitsPerSample}`);
        const header = new ArrayBuffer(44);
        const dv = new DataView(header);
        // 写入 RIFF 块
        this.writeString(dv, 0, 'RIFF');
        let totalDataLen = pcmBuffer.byteLength + 44;
        dv.setUint32(4, totalDataLen, true);
        this.writeString(dv, 8, 'WAVE');
        // 写入 fmt 块
        this.writeString(dv, 12, 'fmt ');
        // fmt块大小
        dv.setUint32(16, 16, true);
        // 格式类别
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
        // 写入data块
        this.writeString(dv, 36, 'data');
        // 数据库大小 
        dv.setUint32(40, pcmBuffer.byteLength, true);

        let resultBuffer = this.concatArrayBuffer(header, pcmBuffer);
        return resultBuffer;
    }
}