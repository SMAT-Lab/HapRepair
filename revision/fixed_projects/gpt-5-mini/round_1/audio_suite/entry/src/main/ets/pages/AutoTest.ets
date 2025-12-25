
/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. All rights reserved.
 */
import { Decimal } from '@kit.ArkTS';
import { media } from '@kit.MediaKit';
import { Logger } from '../utils/Logger';
import { JSON, taskpool, util } from '@kit.ArkTS';
import buffer from '@ohos.buffer';
import { common } from '@kit.AbilityKit';
import { fileUri, ListFileOptions, picker } from '@kit.CoreFileKit';
import fs from '@ohos.file.fs';
import { NodeType, ColorMap, SceneInfo, OutputFile, InputFile, ExpectedOutput } from '../utils/autoTest/Interface'
import{ ExecuteReport, NodeInfo, PipelineInfo, JsonFileInfo, DisplayPipelineInfo } from '../utils/autoTest/Interface'
import { BusinessError } from '@kit.BasicServicesKit';
import { multiPipelineSaveFileBuffer, writeWavFileHeader } from '../utils/ExportFile';
import audioNapi from 'libentry.so';
import { cryptoFramework } from '@kit.CryptoArchitectureKit';

const TAG: string = 'AudioEditTestApp_AudioEdit';
const SUCCESS: number = 0;
const FAILED: number = 1;
const rootPath = `/storage/Users/currentUser/Download/src.main.audiodemo`;
const jsonPath = `${rootPath}/case_json`;
const outputPath = `${rootPath}/output_files`;
const inputDir = `${rootPath}/input_files`;
const expectedOutputPath = `${rootPath}/expected_output`;
const csvReportHeaderText = `执行管线,开始执行时间,结束执行时间,总耗时,输入文件,输入音频时长,输出文件,执行结果\n`;

const jsonSummaryToJsonFileInfoMap: Map<string, JsonFileInfo> = new Map();
const sceneInfoIdToSceneInfoMap: Map<string, SceneInfo> = new Map();
const pipelineIdToPipelineInfoMap: Map<string, PipelineInfo> = new Map();
const nodeInfoIdToNodeInfoMap: Map<string, NodeInfo> = new Map();
const pipelineIdToSingleReportMap: Map<string, ExecuteReport> = new Map();

const dirCheckList: string[] = ["expected_output", "output_files", "input_files", "case_json"];
let preparedPipelineList: string[] = [];

// 初始化pipeline任务子线程向主线程同步消息类
class MultiInitPipelineSonMsg {
  sceneInfoId: string = "";
  pipelineId: string = "";
  outputId: string = "";
  mixerId: string = "";
}

function getNodeTypeFromString(originStr: string): NodeType {
  let ret: NodeType = NodeType.VALID;
  switch (originStr) {
    case "INPUT":
      ret = NodeType.INPUT;
      break;
    case "OUTPUT":
      ret = NodeType.OUTPUT;
      break;
    case "AISS":
      ret = NodeType.AISS;
      break;
    case "NR":
      ret = NodeType.NR;
      break;
    case "VB":
      ret = NodeType.VB;
      break;
    case "MIXER":
      ret = NodeType.MIXER;
      break;
    case "ENV":
      ret = NodeType.ENV;
      break;
    case "EQ":
      ret = NodeType.EQ;
      break;
    case "FIELD":
      ret = NodeType.FIELD;
      break;
  }
  return ret;
}

function getNumberArrayFromString(originStr: string): number[] {
  let ret: number[] = [];
  originStr.replace("[", "").replace("]", "").split(" ").forEach((num) => {
    ret.push(parseInt(num));
  })
  return ret;
}

function parseJsonObject(jsonObjectStr: string): SceneInfo {
  const sceneInfo: SceneInfo = {
    sceneName: '',
    description: '',
    inputFiles: [],
    mixerInfo: [],
    expectedOutput: {
      md5Value: []
    },
    outputFile: {
      sampleRate: 0,
      bitDepth: 0,
      channels: 0,
      format: '',
      nameTemplate: ''
    },
    sceneInfoId: util.generateRandomUUID(true),
    pipelineId: ''
  };
  const expectedOutput: ExpectedOutput = {
    md5Value: []
  }
  sceneInfo.sceneName = Object(jsonObjectStr)["scene_name"];
  sceneInfo.description = Object(jsonObjectStr)["description"];
  sceneInfo.inputFiles = []
  sceneInfo.mixerInfo = []
  let inputLen: number = (Object(jsonObjectStr)["input_files"])["length"];
  let mixerLen: number = (Object(jsonObjectStr)["mixer_info"])["length"];
  for (let i = 0; i < inputLen; i++) {
    const inputFile: InputFile = {
      path: '',
      sampleRate: 0,
      bitDepth: 0,
      channels: 0,
      effect: [],
      inputPathId: '',
      inputId: util.generateRandomUUID(true)
    };
    let originInputFileStr: string = (Object(jsonObjectStr)["input_files"])[i]["path"];
    inputFile.path = originInputFileStr;
    inputFile.inputPathId = util.generateRandomUUID(true);
    inputFile.sampleRate = (Object(jsonObjectStr)["input_files"])[i]["sample_rate"];
    inputFile.bitDepth = (Object(jsonObjectStr)["input_files"])[i]["bit_depth"];
    inputFile.channels = (Object(jsonObjectStr)["input_files"])[i]["channels"];
    inputFile.format = (Object(jsonObjectStr)["input_files"])[i]["format"];
    inputFile.effect = []

    let effectLen: number = (Object(jsonObjectStr)["input_files"])[i]["effect"]["length"];

    for (let j = 0; j < effectLen; j++) {
      let nodeInfo: NodeInfo = {
        type: '',
        gains: [],
        nodeInfoId: util.generateRandomUUID(true)
      }
      nodeInfo.type = (Object(jsonObjectStr)["input_files"])[i]["effect"][j]["type"];
      let mode: number = (Object(jsonObjectStr)["input_files"])[i]["effect"][j]["mode"];
      if (mode != undefined) {
        nodeInfo.mode = mode;
      }
      if ((Object(jsonObjectStr)["input_files"])[i]["effect"][j]["gains"] != undefined) {
        let gains: number[] = getNumberArrayFromString((Object(jsonObjectStr)["input_files"])[i]["effect"][j]["gains"]);
        nodeInfo.gains = gains;
      }
      inputFile.effect.push(nodeInfo);
    }
    sceneInfo.inputFiles.push(inputFile)
  }
  if (!Array.isArray(Object(jsonObjectStr)["expected_output"]["md5Value"])) {
    expectedOutput.md5Value.push((Object(jsonObjectStr)["expected_output"])["md5Value"]);
  } else {
    expectedOutput.md5Value = (Object(jsonObjectStr)["expected_output"])["md5Value"];
  }
  sceneInfo.expectedOutput = expectedOutput;
  for (let i = 0; i < mixerLen; i++) {
    let nodeInfo: NodeInfo = {
      type: '',
      gains: [],
      nodeInfoId: util.generateRandomUUID(true)
    }
    nodeInfo.type = (Object(jsonObjectStr)["mixer_info"])[i]["type"];
    let mode: number = (Object(jsonObjectStr)["mixer_info"])[i]["mode"];
    if (mode != undefined) {
      nodeInfo.mode = mode;
    }
    if ((Object(jsonObjectStr)["mixer_info"])[i]["gains"] != undefined) {
      let gains: number[] = getNumberArrayFromString((Object(jsonObjectStr)["mixer_info"])[i]["gains"]);
      nodeInfo.gains = gains;
    }

    sceneInfo.mixerInfo?.push(nodeInfo);
  }
  sceneInfo.outputFile.nameTemplate = Object(jsonObjectStr)["output_file"]["name_template"];
  sceneInfo.outputFile.format = (Object(jsonObjectStr)["output_file"])["format"];
  sceneInfo.outputFile.sampleRate = (Object(jsonObjectStr)["output_file"])["sample_rate"];
  sceneInfo.outputFile.bitDepth = (Object(jsonObjectStr)["output_file"])["bit_depth"];
  sceneInfo.outputFile.channels = (Object(jsonObjectStr)["output_file"])["channels"];

  return sceneInfo;
}

function getDateStringWithTimeStamp(timestamp: number, mode: number = 0): string {
  let date = new Date(timestamp);
  const year = date.getFullYear();
  const month = ('0' + (date.getMonth() + 1)).slice(-2);
  const day = ('0' + date.getDate()).slice(-2);
  const hour = date.getHours();
  const min = date.getMinutes();
  const sec = date.getSeconds();
  const ms = date.getMilliseconds();
  return mode == 0 ? `${year}/${month}/${day} ${hour}:${min}:${sec}:${ms}` :
    `${year}-${month}-${day} ${hour}-${min}-${sec}`;
}

function getMD5OfFile(filePath: string): string {
  // 获取文件内容
  let file = fs.openSync(filePath, fs.OpenMode.READ_WRITE);
  let buf = new ArrayBuffer(40960);
  fs.readSync(file.fd, buf);
  fs.closeSync(file);

  let mdAlgName = 'SHA256'; // 摘要算法名。
  let md = cryptoFramework.createMd(mdAlgName);
  // 数据量较少时，可以只做一次update，将数据全部传入，接口未对入参长度做限制。
  md.updateSync({ data: new Uint8Array(buf) });
  let mdResult = md.digestSync();
  Logger.info(TAG, '[Sync]:Md result:' + mdResult.data);
  let md5Str = buffer.from(mdResult.data.buffer).toString('hex');
  Logger.info(TAG, '[Sync]:Md string result:' + md5Str);
  return md5Str;
}

function checkMd5IsEqual(outputFileMd5: string[], expectedFileMd5: string[]): boolean {
  let ret = true;
  outputFileMd5.forEach((outputMd5: string) => {
    if (expectedFileMd5.find((expected: string) => expected === outputMd5) == undefined) {
      ret = false;
    }
  })
  return ret;
}

@Concurrent
async function initAllInputNode(jsonFileInfo:JsonFileInfo, sceneInfoIdToSceneInfoMap: Map<string, SceneInfo>, pipelineIdToPipelineInfoMap:Map<string, PipelineInfo>): Promise<number> {
  const TAG: string = 'AudioEditTestApp_AudioEdit';
  const inputDir = `/storage/Users/currentUser/Download/src.main.audiodemo/input_files`;
  const SUCCESS: number = 0;
  const FAILED: number = 1;
  Logger.info(TAG, `initMultiPipelineTask start`);

  for (const sceneInfoId of jsonFileInfo.sceneInfos) {
    const sceneInfo = sceneInfoIdToSceneInfoMap.get(sceneInfoId);
    const pipelineId = sceneInfo?.pipelineId;
    if (sceneInfo == undefined || pipelineId == undefined) {
      Logger.error(TAG, `initAllInputNode get null sceneInfo`);
      return FAILED;
    }
    const pipelineInfo = pipelineIdToPipelineInfoMap.get(pipelineId);
    if (pipelineInfo == undefined) {
      Logger.error(TAG, `initAllInputNode get null pipelineInfo`);
      return FAILED;
    }
    let status = audioNapi.multiPipelineEnvPrepare(pipelineId);
    if (status != SUCCESS) {
      Logger.error(TAG, `multiPipelineEnvPrepare ERROR while initAllInputNode`);
      return FAILED;
    }
    for (const inputFile of sceneInfo.inputFiles) {
    // 获取文件内容
    let inputPath = `${inputDir}/${inputFile.path}`;
    let file = fs.openSync(inputPath, fs.OpenMode.READ_WRITE);
    let fsStat = fs.statSync(inputPath);
    let buffer = new ArrayBuffer(fsStat.size);
    fs.readSync(file.fd, buffer);
    Logger.info(TAG, `input FIle: ${inputPath}, buffer.length: ${buffer.byteLength}`);
    let inputNodeStatus =
      audioNapi.multiAudioInAndOutInit(inputFile.inputId, pipelineInfo.outputId, pipelineInfo.mixerId, file.fd,
        buffer.byteLength);
    fs.closeSync(file.fd);
    if (inputNodeStatus != SUCCESS) {
      Logger.error(TAG, `input node init ERROR, inputId:${inputFile.inputId}, pipelineId :${pipelineId}`);
      return FAILED;
    }
    Logger.info(TAG, `input node init done, inputId:${inputFile.inputId}, pipelineId :${pipelineId}`);
    }
  Logger.info(TAG, `all input node had inited , pipelineId:${pipelineId}`);
  }
  return SUCCESS;
}

@Concurrent
async function multiInitPipeline(sceneInfoId: string, pipelineId: string): Promise<number> {
  const TAG: string = 'AudioEditTestApp_AudioEdit';
  Logger.info(TAG, ' AutoTest multiInitPipeline start');
  let ret = audioNapi.audioEditNodeInitMultiPipeline(pipelineId);
  let msg: MultiInitPipelineSonMsg = {
    sceneInfoId: sceneInfoId,
    pipelineId: pipelineId,
    outputId: util.generateRandomUUID(true),
    mixerId: util.generateRandomUUID(true)
  }
  taskpool.Task.sendData(msg);
  Logger.info(TAG, ' AutoTest multiInitPipeline end');
  return ret;
}

async function noticePipelineTask(msg: MultiInitPipelineSonMsg, preparedPipelineListParam: string[], pipelineIdToPipelineInfoMapParam: Map<string, PipelineInfo>) {
  Logger.info(TAG, `noticePipelineTask start msg: ${JSON.stringify(msg)}`);
  const sceneInfo = sceneInfoIdToSceneInfoMap.get(msg.sceneInfoId);
  if (sceneInfo === undefined) {
    return;
  }
  let pipelineInfo: PipelineInfo = {
    outputId: msg.outputId,
    mixerId: msg.mixerId,
    inputIdList: [],
    pipelineId: msg.pipelineId,
    multiRenderFrameFlag: false,
    sceneInfoId: msg.sceneInfoId
  }
  sceneInfo.pipelineId = msg.pipelineId;
  preparedPipelineListParam.push(msg.pipelineId);
  pipelineIdToPipelineInfoMapParam.set(msg.pipelineId, pipelineInfo);
  Logger.info(TAG, `noticePipelineTask end`);
}

@Concurrent
async function multiInitInput(inputId: string, outputId: string, mixerId: string, pipelineId: string, audioFile: string,
  multiPipelineFlag: boolean): Promise<number> {
  const TAG: string = 'AudioEditTestApp_AudioEdit';
  Logger.info(TAG, ` AutoTest multiInitInput start, inputId:${inputId}, pipelineId:${pipelineId}`);
  const inputDir = `/storage/Users/currentUser/Download/src.main.audiodemo/input_files`;

  // 获取文件内容
  let inputPath = `${inputDir}/${audioFile}`;
  let file = fs.openSync(inputPath, fs.OpenMode.READ_WRITE);
  let fsStat = fs.statSync(inputPath);
  let buffer = new ArrayBuffer(fsStat.size);
  fs.readSync(file.fd, buffer);

  Logger.info(TAG, `input FIle: ${inputPath}, buffer.length: ${buffer.byteLength}`);

  const wavView = new Uint8Array(buffer);
  let wavHeaderBuffer = wavView.slice(0, 44).buffer;
  const dataView = new DataView(wavHeaderBuffer);
  // 获取通道数, 通道数位于22字节
  const channels: number = dataView.getUint16(22, true);
  // 获取采样率, 采样率位于第 24 到28 字节
  const sampleRate: number = dataView.getUint32(24, true);
  // 获取位深，位深位于 34字节
  const bitsPerSample: number = dataView.getUint16(34, true);
  // 获取音频的长度
  const fmtSize = dataView.getUint32(16, true); // fmt chunk size
  let offset = 20 + fmtSize;
  let pcmLength = 0;
  while (offset <= buffer.byteLength) {
    const chunkId = dataView.getUint32(offset, true);
    offset += 4;
    const chunkSize = dataView.getUint32(offset, true);
    offset += 4;
    // 'data' 在 ASCII 小端字节
    if (chunkId === 0x61746164) {
      // 找到了 data 块，返回其大小
      pcmLength = chunkSize;
      break;
    }
    offset += chunkSize;
  }
  Logger.info(TAG, `pcmLength: ${pcmLength}`);
  Logger.info(TAG,
    `audioEffectNodeTestSuccess01 sampleRate: ${sampleRate}, bitsPerSample: ${bitsPerSample}, channels: ${channels}`);
  if (multiPipelineFlag) {
    audioNapi.multiPipelineEnvPrepare(pipelineId);
  }
  let ret = audioNapi.multiAudioInAndOutInit(inputId, outputId, mixerId, file.fd, buffer.byteLength);
  fs.closeSync(file.fd);
  Logger.info(TAG, ` AutoTest multiInitInput end, inputId:${inputId}, pipelineId:${pipelineId}`);
  return ret;
}

@Concurrent
async function multiInitEffect(inputId: string | undefined, nodeType: NodeType, effect: NodeInfo,
  pipelineId: string, multiPipelineFlag: boolean): Promise<number> {
  const TAG: string = 'AudioEditTestApp_AudioEdit';
  Logger.info(TAG, ' AutoTest multiInitEffect start');
  let ret: number = 0;

  if (multiPipelineFlag) {
    audioNapi.multiPipelineEnvPrepare(pipelineId);
  }
  switch (nodeType) {
    case NodeType.NR:
      let NRId: string = util.generateRandomUUID(true);
      let ret: number = audioNapi.multiAddNoiseReduction(NRId, inputId);
      if (ret != 0) {
        Logger.error(TAG, `addNoiseReduction error number is ${ret}`);
      } else {
        Logger.info(TAG, `addNoiseReduction node successful!!!`);
      }
      break;
    case NodeType.AISS:
      let AISSId: string = util.generateRandomUUID(true);
      ret = audioNapi.multiAddAudioSeparation(effect.mode, AISSId, inputId);
      if (ret != 0) {
        Logger.error(TAG, `addAudioSeparation error number is ${ret}`);
      } else {
        Logger.info(TAG, `addAudioSeparation node successful!!!`);
      }
      taskpool.Task.sendData(pipelineId);
      break;
    case NodeType.ENV:
      let ENVId: string = util.generateRandomUUID(true);
      ret = audioNapi.multiStartEnvEffect(inputId, ENVId, effect.mode);
      if (ret != 0) {
        Logger.error(TAG, `multiStartEnvEffect error number is ${ret}`);
      } else {
        Logger.info(TAG, `multiStartEnvEffect node successful!!!`);
      }
      break;
    case NodeType.FIELD:
      let FIELDId: string = util.generateRandomUUID(true);
      ret = audioNapi.multiStartFieldEffect(inputId, effect.mode, FIELDId);
      if (ret != 0) {
        Logger.error(TAG, `startFieldEffect error number is ${ret}`);
      } else {
        Logger.info(TAG, `startFieldEffect node successful!!!`);
      }
      break;
    case NodeType.VB:
      let VBId: string = util.generateRandomUUID(true);
      ret = audioNapi.multiStartVBEffect(inputId, effect.mode, VBId);
      if (ret != 0) {
        Logger.error(TAG, `startVBEffect error number is ${ret}`);
      } else {
        Logger.info(TAG, `startVBEffect node successful!!!`);
      }
      break;
    case NodeType.EQ:
      let EQId: string = util.generateRandomUUID(true);
      if (effect.gains == undefined || effect.gains.length == 0) {
        ret = audioNapi.multiSetEqualizerMode(effect.mode, EQId, inputId);
      } else {
        ret = audioNapi.multiSetEqualizerFrequencyBandGains(effect.gains, EQId, inputId);
      }
      if (ret != 0) {
        Logger.error(TAG, `startVBEffect error number is ${ret}`);
      } else {
        Logger.info(TAG, `startVBEffect node successful!!!`);
      }
      break;
  }
  Logger.info(TAG, ' AutoTest multiInitEffect end');
  return ret;
}

async function noticeEffectTask(msg: string, pipelineMap: Map<string, PipelineInfo>) {
  Logger.info(TAG, `noticeEffectTask start pipelineId: ${JSON.stringify(msg)}`);
  const pipelineInfo = pipelineMap.get(msg);
  if (pipelineInfo === undefined) {
    return;
  }
  pipelineInfo.multiRenderFrameFlag = true;
  Logger.info(TAG, `noticePipelineTask end`);
}

@Builder
export function AutoTestBuilder() {
  AutoTest()
}

@Component
export struct AutoTest {
  @State text: string = "请选择预定义场景";
  @State index: number = 0;
  @State selectedFileSummary: string = "";
  @State selectOptionList: SelectOption[] = [];
  @State nodeInfoShow: boolean = false;
  choosedNode: NodeInfo = {
    nodeInfoId: '',
    type: ''
  }
  @State displayPipelineInfoList: DisplayPipelineInfo[] = [];
  pageInfos: NavPathStack = new NavPathStack();

  async aboutToAppear(): Promise<void> {
    Logger.info(TAG, 'AutoTest aboutToAppear');

    // 初始化支持自动化测试运行的文件夹
    try {
      let fileList: string[] = fs.listFileSync(rootPath);
      dirCheckList.forEach((check: string) => {
        if (!fileList.find((ele: string) => ele != "init.txt" && ele === check)) {
          Logger.info(TAG, `dir:${check} not existed`);
          let dirPath = `${rootPath}/${check}`;
          fs.mkdirSync(dirPath);
        }
      })
    } catch (e) {
      Logger.error(TAG, "rootPath is null, need init rootPath");
      await this.initAutoTestDir();
    }

    const dirPath = new fileUri.FileUri(rootPath + `/case_json`).path;
    let listFileOption: ListFileOptions = {
      filter: {
        suffix: [".json"]
      }
    };
    let fileList = fs.listFileSync(dirPath, listFileOption);
    fileList.forEach((jsonFile: string) => {
      let jsonFileInfo: JsonFileInfo = {
        jsonInfoId: util.generateRandomUUID(true),
        filePath: '',
        sceneInfos: [],
        summary: ''
      }

      const jsonFilePath = `${rootPath}/case_json/${jsonFile}`
      jsonFileInfo.filePath = jsonFilePath;
      try {
        let file = fs.openSync(jsonFilePath, fs.OpenMode.READ_WRITE);
        let fsStat = fs.statSync(jsonFilePath);
        let buf = new ArrayBuffer(fsStat.size);
        fs.readSync(file.fd, buf);
        let fileData: string = buffer.from(buf).toString();
        let obj = JSON.parse(fileData);
        jsonFileInfo.summary = Object(obj)["summary"];
        this.selectOptionList.push({ value: jsonFileInfo.summary });
        jsonSummaryToJsonFileInfoMap.set(jsonFileInfo.summary, jsonFileInfo);
        const caseNum: number = Object(obj)["scenes"]["length"];
        if (caseNum > 0) {
          for (let i = 0; i < caseNum; i++) {
            const sceneInfo: SceneInfo = parseJsonObject(Object(obj)["scenes"][i]);
            sceneInfoIdToSceneInfoMap.set(sceneInfo.sceneInfoId, sceneInfo);
            jsonFileInfo.sceneInfos.push(sceneInfo.sceneInfoId);
          }
        }
      } catch (e) {
        Logger.error(TAG, `Open file:${jsonFilePath} Failed, error info:${e}`)
      }
    })
    if (jsonSummaryToJsonFileInfoMap.size > 0 && sceneInfoIdToSceneInfoMap.size > 0) {
      Logger.info(TAG, "Parse json file success");
    } else {
      Logger.error(TAG, "Parse json file failed");
    }
  }

  aboutToDisappear(): void {
    Logger.info(TAG, 'aboutToDisappear');
    // 销毁管线，销毁引擎
    this.destroyPipeline();
  }

  build() {
    NavDestination() {
      Flex({ justifyContent: FlexAlign.Start, alignItems: ItemAlign.Center }) {
        Column() {
          Text('选择场景：')
        }.width('30%')
      }.height('5%')

      Row() {
        Select(this.selectOptionList)
          .width('100%')
          .selected(this.index)
          .value(this.text)
          .font({ size: 16, weight: 500 })
          .fontColor('#182431')
          .selectedOptionFont({ size: 16, weight: 400 })
          .optionFont({ size: 16, weight: 400 })
          .menuAlign(MenuAlignType.START, { dx: 0, dy: 0 })
          .optionWidth(350)
          .optionHeight(300)
          .onSelect((index: number, text: string) => {
            this.index = index;
            this.text = text
            this.selectedFileSummary = text;
            const sceneInfos = jsonSummaryToJsonFileInfoMap.get(text)?.sceneInfos;
            if (sceneInfos == undefined) {
              return;
            }
            this.displayPipelineInfoList = sceneInfos.map((sceneInfoId: string): DisplayPipelineInfo => {
              const sceneInfo = sceneInfoIdToSceneInfoMap.get(sceneInfoId);
              const displayPipelineInfo: DisplayPipelineInfo = {
                effectChain: new Map(),
                mixerChain: [],
                inputNodeIdList: []
              }
              if (sceneInfo == undefined) {
                return displayPipelineInfo;
              }

              sceneInfo?.inputFiles?.forEach((inputFile: InputFile) => {
                let inputNode: NodeInfo = {
                  nodeInfoId: util.generateRandomUUID(true),
                  type: "INPUT",
                  format: inputFile.format,
                  sampleRate: inputFile.sampleRate,
                  bitDepth: inputFile.bitDepth,
                  channels: inputFile.channels
                }
                nodeInfoIdToNodeInfoMap.set(inputNode.nodeInfoId, inputNode);
                displayPipelineInfo.inputNodeIdList.push(inputNode.nodeInfoId);
                displayPipelineInfo.effectChain.set(inputNode.nodeInfoId, []);
                inputFile.effect.forEach((effectNode: NodeInfo) => {
                  nodeInfoIdToNodeInfoMap.set(effectNode.nodeInfoId, effectNode);
                  displayPipelineInfo.effectChain.get(inputNode.nodeInfoId)?.push(effectNode.nodeInfoId);
                })
              })
              sceneInfo?.mixerInfo?.forEach((mixerNode: NodeInfo) => {
                nodeInfoIdToNodeInfoMap.set(mixerNode.nodeInfoId, mixerNode);
                displayPipelineInfo.mixerChain.push(mixerNode.nodeInfoId)
              })

              let outputNode: NodeInfo = {
                nodeInfoId: util.generateRandomUUID(true),
                type: "OUTPUT",
                format: sceneInfo.outputFile.format,
                sampleRate: sceneInfo.outputFile.sampleRate,
                bitDepth: sceneInfo.outputFile.bitDepth,
                channels: sceneInfo.outputFile.channels
              }
              nodeInfoIdToNodeInfoMap.set(outputNode.nodeInfoId, outputNode);
              displayPipelineInfo.mixerChain.push(outputNode.nodeInfoId)
              return displayPipelineInfo;
            })
          })
      }.height('5%')


      Column() {
        if (this.displayPipelineInfoList.length > 0) {
          List().width('100%').height('100%') {
            ForEach(this.displayPipelineInfoList, (displayPipelineInfo: DisplayPipelineInfo) => {
              ListItem() {
                this.PipelineGraphArea(displayPipelineInfo)
              }
            }, (item, index) => item)
          }
        }
      }.height('75%').width('100%').padding(2)

      // .border({ width: '2vp', color: Color.Pink, radius: '10vp' })

      Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceAround, alignItems: ItemAlign.Center }) {
        Button() {
          Text(`执行选中用例`)
        }
        .onClick(() => {
          this.executeSingleFile(`${this.selectedFileSummary}`);
        })
        .backgroundColor(0x317aff)
        .borderRadius(10)
        .width(100)
        .height(50)

        Button() {
          Text(`执行全部用例`)
        }
        .onClick(() => {
          this.executeAllFile();
        })
        .backgroundColor(0x317aff)
        .borderRadius(10)
        .width(100)
        .height(50)
        .margin({ left: 20 })
      }.height('20%').width('100%')

    }
    .onReady((context: NavDestinationContext) => {
      this.pageInfos = context.pathStack;
    })
    .title('自动化测试')
    .height('100%')
    .width('100%')
  }

  @Builder
  PipelineGraphArea(displayPipelineInfo: DisplayPipelineInfo) {
    Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.SpaceAround }) {
      this.EffectGraphArea(displayPipelineInfo.inputNodeIdList, displayPipelineInfo.effectChain)
      this.MixerGraphArea(displayPipelineInfo.mixerChain)
    }
    .border({ width: '3vp', color: Color.Blue, radius: '10vp' })
    .padding({ left: '10vp', right: '10vp' })
    .margin({ bottom: '10vp' })
  }

  @Builder
  EffectGraphArea(inputNodeList: string[], effectChainMap: Map<string, string[]>) {
    Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.SpaceAround }) {
      if (this.displayPipelineInfoList.length > 0) {

        Text("Effect Area")
        
      }
      Column() {
        ForEach(inputNodeList, (inputNodeId: string) => {
          Row() {
            List().width('100%').height('100%') {
              ListItem() {
                this.NodeButton(this.makeNodeMapNotBlank(inputNodeId))
              }

              ForEach(effectChainMap.get(inputNodeId), (effectNodeId: string) => {
                ListItem() {
                  this.NodeButton(this.makeNodeMapNotBlank(effectNodeId))
                }
              }, (item, index) => item)
            }
            .width('100%')
            .listDirection(Axis.Horizontal)
            .alignListItem(ListItemAlign.Center)
            .focusWrapMode(FocusWrapMode.WRAP_WITH_ARROW)
            .scrollBar(BarState.Off)
            .margin({ bottom: inputNodeList.length > 1 ? 10 : 0 })
          }
        }, (item, index) => item)
      }
      .border({ width: '1vp', color: Color.Blue, radius: '10vp' })
      .backgroundColor(Color.Gray)

    }
  }

  @Builder
  MixerGraphArea(mixerNodeChain: string[]) {
    Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.SpaceAround }) {
      if (this.displayPipelineInfoList.length > 0) {

        Text("Output Area")
        
      }

      Row() {
        List().width('100%').height('100%') {
          ForEach(mixerNodeChain, (nodeId: string) => {
            ListItem() {
              this.NodeButton(this.makeNodeMapNotBlank(nodeId));
            }
          }, (item, index) => item)
        }
        .width('100%')
        .listDirection(Axis.Horizontal)
        .alignListItem(ListItemAlign.Center)
        .focusWrapMode(FocusWrapMode.WRAP_WITH_ARROW)
        .scrollBar(BarState.Off)
        .border({ width: '1vp', color: Color.Blue, radius: '10vp' })
        .backgroundColor(Color.Gray)
      }
    }
    .margin({ bottom: 5 })

  }

  @Builder
  NodeButton(nodeInfo: NodeInfo) {
    Column() {
      Button() {
        Text(`${getNodeTypeFromString(nodeInfo.type)}`)
          .wordBreak(WordBreak.BREAK_WORD)
          .lineBreakStrategy(LineBreakStrategy.BALANCED)
      }
      .backgroundColor(ColorMap.get(getNodeTypeFromString(nodeInfo.type)))
      .borderRadius(10)
      .width(60)
      .height(60)
      .margin({ left: 10 })
    }
    .onClick(() => {
      this.nodeInfoShow = true;
    })
    .bindSheet(this.nodeInfoShow, this.NodeInfoDialog(nodeInfo), {
      height: '50%',
      backgroundColor: Color.White,
      maskColor: Color.Transparent,
      showClose: false,
      onWillAppear: () => {
      },
      onAppear: () => {
      },
      onWillDisappear: () => {
      },
      onDisappear: () => {
        this.nodeInfoShow = false;
      },
      onWillDismiss: (DismissSheetAction: DismissSheetAction) => {
        if (DismissSheetAction.reason != DismissReason.TOUCH_OUTSIDE) {
          DismissSheetAction.dismiss()
        }
      }
    })
  }

  @Builder
  NodeInfoDialog(nodeInfo: NodeInfo) {
    Column() {
      Row() {
        Column() {
          Image($r('app.media.close_window'))
            .width(20)
            .height(20)
        }
        .width('20%')
        .onClick(() => {
          this.nodeInfoShow = false;
        })

        Column() {
          Text('Node Info').fontSize(20)
        }.width('80%')
      }

      Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.SpaceAround }) {
        if (nodeInfo.type) {
          Text(`node_type: ${nodeInfo.type}`).fontSize(20)
        }
        if (nodeInfo.mode) {
          Text(`mode: ${nodeInfo.mode}`).fontSize(20)
        }
        if (nodeInfo.gains) {
          Text(`gains: ${nodeInfo.gains.toString()}`).fontSize(20)
        }
        if (nodeInfo.format) {
          Text(`format: ${nodeInfo.format}`).fontSize(20)
        }
        if (nodeInfo.sampleRate) {
          Text(`sampleRate: ${nodeInfo.sampleRate}`).fontSize(20)
        }
        if (nodeInfo.channels) {
          Text(`channels: ${nodeInfo.channels}`).fontSize(20)
        }
        if (nodeInfo.bitDepth) {
          Text(`bitDepth: ${nodeInfo.bitDepth}`).fontSize(20)
        }
      }.height('80%')
    }

  }

  async executeAllFile() {
    Logger.info(TAG, "executeAllFile start");
    const jsonFiles = Array.from(jsonSummaryToJsonFileInfoMap.values());
    for (const jsonFile of jsonFiles) {
      Logger.info(TAG, `before task jsonFile=${jsonFile.summary}`);
      let ret = await this.executeSingleFile(jsonFile.summary, false);
      Logger.info(TAG, `before task ret = ${ret}`);
    }
    await this.saveAllCaseExecuteReport();
    this.destroyPipeline();
    Logger.info(TAG, "executeAllFile end");
  }

  // async executeSingleFile(fileSummary: string, reportFlag: boolean = true): Promise<number> {
  //   if (jsonFileInfo == undefined) {
  //     return 1;
  //   } else if (jsonFileInfo.sceneInfos.length >= 1) {
  //     let scenePromises = jsonFileInfo.sceneInfos.map(async (sceneInfoId: string) => {
  //       if (sceneInfo == undefined) {
  //         return;
  //       }
  //       const pipelineInfo: PipelineInfo = {
  //         pipelineId: pipelineId,
  //         inputIdList: [],
  //         outputId: '',
  //         mixerId: '',
  //         multiRenderFrameFlag: false,
  //         sceneInfoId: ''
  //       }
  //       const singleReport: ExecuteReport = {
  //         sceneInfoId: jsonFileInfo.sceneInfos[0],
  //         sceneName: sceneInfo.sceneName,
  //         startTime: getDateStringWithTimeStamp(start),
  //         startTimeStamp: start,
  //         endTime: '',
  //         endTimeStamp: 0,
  //         costTime: 0,
  //         result: false,
  //         exeReportPath: '',
  //       }
  //
  //       const buffer: object =
  //     })
  //     await Promise.all(scenePromises).then(async () => {
  //       if (reportFlag) {
  //       }
  //     })
  //   }
  //   return 0;
  // }

  async executeSingleFile(fileSummary: string, reportFlag: boolean = true): Promise<number> {
    Logger.info(TAG, `executeSingleFile start fileSummary:${fileSummary}`);
    const jsonFileInfo = jsonSummaryToJsonFileInfoMap.get(fileSummary);
    if (jsonFileInfo == undefined) {
      return 1;
    } else if (jsonFileInfo.sceneInfos.length >= 1) {
      let status = this.initAllPipeline(jsonFileInfo);
      if (status != SUCCESS) {
        Logger.error(TAG, `initAllPipeline ERROR`);
        return FAILED;
      }
      let taskExeStatus = await taskpool.execute(initAllInputNode, jsonFileInfo, sceneInfoIdToSceneInfoMap, pipelineIdToPipelineInfoMap);
      if (taskExeStatus != SUCCESS) {
        Logger.error(TAG, `initAllInputNode ERROR`);
        return FAILED;
      }
      status = this.initAllEffectNode(jsonFileInfo);
      if (status != SUCCESS) {
        Logger.error(TAG, `initAllEffectNode ERROR`);
        return FAILED;
      }
      Logger.info(TAG, `initAllEffectNode successed, ready to render frame`);
      let scenePromises = jsonFileInfo.sceneInfos.map(async (sceneInfoId: string) => {
        const sceneInfo = sceneInfoIdToSceneInfoMap.get(sceneInfoId);
        const pipelineId = sceneInfo?.pipelineId;
        if (sceneInfo == undefined || pipelineId == undefined) {
          Logger.error(TAG, `executeSingleFile get null sceneInfo`);
          return FAILED;
        }
        const pipelineInfo = pipelineIdToPipelineInfoMap.get(pipelineId);
        if (pipelineInfo == undefined) {
          Logger.error(TAG, `executeSingleFile get null pipelineInfo`);
          return FAILED;
        }

        let outputInfo = sceneInfo.outputFile;
        const multiRenderFrameFlag = pipelineInfo.multiRenderFrameFlag;
        const buffer: object =
          await taskpool.execute(multiPipelineSaveFileBuffer, pipelineId, outputInfo, multiRenderFrameFlag);
        await this.saveSingleFile(pipelineId, buffer as ArrayBuffer[], ".wav", outputInfo, multiRenderFrameFlag)
        this.generateSingleReportInfo(pipelineId);
        return SUCCESS;
      })
      await Promise.all(scenePromises).then(async () => {
        Logger.info(TAG, `scenePromises done `);
        if (reportFlag) {
          await this.saveSingleCaseExecuteReport(fileSummary);
        }
        this.destroyPipeline();
      })
    }
    return 0;
  }

  async initMultiPipelineTask(sceneInfoId: string, pipelineId: string) {
    Logger.info(TAG, `initMultiPipelineTask start`);
    const sceneInfo = sceneInfoIdToSceneInfoMap.get(sceneInfoId);
    if (sceneInfo == undefined) {
      Logger.error(TAG, `initMultiPipelineTask get null info of :${sceneInfoId}`)
      return;
    }
    let pipelineTask: taskpool.Task = new taskpool.Task(multiInitPipeline, sceneInfo.sceneInfoId, pipelineId);
    pipelineTask.onReceiveData((msg) => noticePipelineTask(msg, preparedPipelineList, pipelineIdToPipelineInfoMap));
    await taskpool.execute(pipelineTask);
    for (const ele of sceneInfo.inputFiles){
      let pipelineId: string = sceneInfo.pipelineId;
      const inputId = ele.inputId;
      Logger.info(TAG, `input node init start, inputId:${inputId}, pipelineId :${pipelineId}`);
      await taskpool.execute(multiInitInput, inputId, pipelineIdToPipelineInfoMap.get(pipelineId)?.outputId,
        pipelineIdToPipelineInfoMap.get(pipelineId)?.mixerId, pipelineId, ele.path, true)
      pipelineIdToPipelineInfoMap.get(pipelineId)?.inputIdList.push(inputId);
      Logger.info(TAG, `input node init done, inputId:${inputId}, pipelineId :${pipelineId}`);
    }
    Logger.info(TAG, `all input node had inited , pipelineId:${pipelineId}`);
    for (const ele of sceneInfo.inputFiles){
      const inputId = ele.inputId;
      for (const effect of ele.effect) {
        Logger.info(TAG, `effect node init start, inputId :${inputId}, pipelineId:${pipelineId}`);
        let nodeType: NodeType = getNodeTypeFromString(effect.type);
        let multiInitEffectTask: taskpool.Task =
          new taskpool.Task(multiInitEffect, inputId, nodeType, effect, pipelineId, true);
        multiInitEffectTask.onReceiveData(noticeEffectTask);
        await taskpool.execute(multiInitEffectTask);
        Logger.info(TAG, `effect node init end, inputId :${inputId}, pipelineId:${pipelineId}`);
      }
    }
    Logger.info(TAG, `all multiInitEffect end, pipelineId:${pipelineId}`);
    return Promise.resolve();
  }

  initAllPipeline(jsonFileInfo:JsonFileInfo): number {
    Logger.info(TAG, `initAllPipeline start`);
    for (const sceneInfoId of jsonFileInfo.sceneInfos) {
      const sceneInfo = sceneInfoIdToSceneInfoMap.get(sceneInfoId);
      if (sceneInfo == undefined) {
        return FAILED;
      }
      const pipelineId = util.generateRandomUUID(true);
      const pipelineInfo: PipelineInfo = {
        pipelineId: pipelineId,
        inputIdList: [],
        outputId: util.generateRandomUUID(true),
        mixerId: util.generateRandomUUID(true),
        multiRenderFrameFlag: false,
        sceneInfoId: sceneInfoId
      }
      const start = Date.now();
      const singleReport: ExecuteReport = {
        sceneInfoId: jsonFileInfo.sceneInfos[0],
        sceneName: sceneInfo.sceneName,
        startTime: getDateStringWithTimeStamp(start),
        startTimeStamp: start,
        endTime: '',
        endTimeStamp: 0,
        costTime: 0,
        result: false,
        exeReportPath: '',
      }
      pipelineIdToSingleReportMap.set(pipelineId, singleReport);
      pipelineIdToPipelineInfoMap.set(pipelineId, pipelineInfo);
      sceneInfo.pipelineId = pipelineId;
      let pipelineStatus = audioNapi.audioEditNodeInitMultiPipeline(pipelineId)
      if (pipelineStatus != SUCCESS) {
        Logger.error(TAG, `pipeline init ERROR, pipelineId :${pipelineId}`);
        return FAILED;
      }
      sceneInfo.pipelineId = pipelineId;
      preparedPipelineList.push(pipelineId);
      pipelineIdToPipelineInfoMap.set(pipelineId, pipelineInfo);
      Logger.info(TAG, `pipeline inited, pipelineId: ${pipelineId}`);
    }
    Logger.info(TAG, `all pipeline inited`);
    return SUCCESS;
  }

  async initAllInputNode(jsonFileInfo:JsonFileInfo): Promise<number> {
    Logger.info(TAG, `initMultiPipelineTask start`);
    for (const sceneInfoId of jsonFileInfo.sceneInfos) {
      const sceneInfo = sceneInfoIdToSceneInfoMap.get(sceneInfoId);
      const pipelineId = sceneInfo?.pipelineId;
      if (sceneInfo == undefined || pipelineId == undefined) {
        Logger.error(TAG, `initAllInputNode get null sceneInfo`);
        return FAILED;
      }
      const pipelineInfo = pipelineIdToPipelineInfoMap.get(pipelineId);
      if (pipelineInfo == undefined) {
        Logger.error(TAG, `initAllInputNode get null pipelineInfo`);
        return FAILED;
      }
      let status = audioNapi.multiPipelineEnvPrepare(pipelineId);
      if (status != SUCCESS) {
        Logger.error(TAG, `multiPipelineEnvPrepare ERROR while initAllInputNode`);
        return FAILED;
      }
      for (const inputFile of sceneInfo.inputFiles) {
        // 获取文件内容
        let inputPath = `${inputDir}/${inputFile.path}`;
        let file = fs.openSync(inputPath, fs.OpenMode.READ_WRITE);
        let fsStat = fs.statSync(inputPath);
        let buffer = new ArrayBuffer(fsStat.size);
        fs.readSync(file.fd, buffer);

        Logger.info(TAG, `input FIle: ${inputPath}, buffer.length: ${buffer.byteLength}`);

        const wavView = new Uint8Array(buffer);
        let wavHeaderBuffer = wavView.slice(0, 44).buffer;
        const dataView = new DataView(wavHeaderBuffer);
        // 获取通道数, 通道数位于22字节
        const channels: number = dataView.getUint16(22, true);
        // 获取采样率, 采样率位于第 24 到28 字节
        const sampleRate: number = dataView.getUint32(24, true);
        // 获取位深，位深位于 34字节
        const bitsPerSample: number = dataView.getUint16(34, true);
        // 获取音频的长度
        const fmtSize = dataView.getUint32(16, true); // fmt chunk size
        let offset = 20 + fmtSize;
        let pcmLength = 0;
        while (offset <= buffer.byteLength) {
          const chunkId = dataView.getUint32(offset, true);
          offset += 4;
          const chunkSize = dataView.getUint32(offset, true);
          offset += 4;
          // 'data' 在 ASCII 小端字节
          if (chunkId === 0x61746164) {
            // 找到了 data 块，返回其大小
            pcmLength = chunkSize;
            break;
          }
          offset += chunkSize;
        }
        Logger.info(TAG, `pcmLength: ${pcmLength}`);
        Logger.info(TAG,
          `audioEffectNodeTestSuccess01 sampleRate: ${sampleRate}, bitsPerSample: ${bitsPerSample}, channels: ${channels}`);
        let inputNodeStatus =
          audioNapi.multiAudioInAndOutInit(inputFile.inputId, pipelineInfo.outputId, pipelineInfo.mixerId, file.fd,
            buffer.byteLength);
        fs.closeSync(file.fd);
        if (inputNodeStatus != SUCCESS) {
          Logger.error(TAG, `input node init ERROR, inputId:${inputFile.inputId}, pipelineId :${pipelineId}`);
          return FAILED;
        }
        Logger.info(TAG, `input node init done, inputId:${inputFile.inputId}, pipelineId :${pipelineId}`);
      }
      Logger.info(TAG, `all input node had inited , pipelineId:${pipelineId}`);
    }
    return SUCCESS;
  }

  initAllEffectNode(jsonFileInfo:JsonFileInfo): number {
    Logger.info(TAG, `initAllEffectNode start`);
    for (const sceneInfoId of jsonFileInfo.sceneInfos) {
      const sceneInfo = sceneInfoIdToSceneInfoMap.get(sceneInfoId);
      const pipelineId = sceneInfo?.pipelineId;
      if (sceneInfo == undefined || pipelineId == undefined) {
        Logger.error(TAG, `initAllEffectNode get null sceneInfo`);
        return FAILED;
      }
      const pipelineInfo = pipelineIdToPipelineInfoMap.get(pipelineId);
      if (pipelineInfo == undefined) {
        Logger.error(TAG, `initAllEffectNode get null pipelineInfo`);
        return FAILED;
      }
      let status = audioNapi.multiPipelineEnvPrepare(pipelineId);
      if (status != SUCCESS) {
        Logger.error(TAG, `multiPipelineEnvPrepare ERROR while initAllInputNode`);
        return FAILED;
      }
      for (const inputFile of sceneInfo.inputFiles) {
        const inputId = inputFile.inputId;
        for (const effect of inputFile.effect) {
          const nodeType: NodeType = getNodeTypeFromString(effect.type);
          switch (nodeType) {
            case NodeType.NR:
              let NRId: string = util.generateRandomUUID(true);
              let status: number = audioNapi.multiAddNoiseReduction(NRId, inputId);
              if (status != SUCCESS) {
                Logger.error(TAG, `addNoiseReduction error number is ${status}`);
                return FAILED;
              } else {
                Logger.info(TAG, `addNoiseReduction node successful!!!`);
              }
              break;
            case NodeType.AISS:
              let AISSId: string = util.generateRandomUUID(true);
              status = audioNapi.multiAddAudioSeparation(effect.mode, AISSId, inputId);
              if (status != SUCCESS) {
                Logger.error(TAG, `addAudioSeparation error number is ${status}`);
                return FAILED;
              } else {
                Logger.info(TAG, `addAudioSeparation node successful!!!`);
              }
              pipelineInfo.multiRenderFrameFlag = true
              break;
            case NodeType.ENV:
              let ENVId: string = util.generateRandomUUID(true);
              status = audioNapi.multiStartEnvEffect(inputId, ENVId, effect.mode);
              if (status != SUCCESS) {
                Logger.error(TAG, `multiStartEnvEffect error number is ${status}`);
                return FAILED;
              } else {
                Logger.info(TAG, `multiStartEnvEffect node successful!!!`);
              }
              break;
            case NodeType.FIELD:
              let FIELDId: string = util.generateRandomUUID(true);
              status = audioNapi.multiStartFieldEffect(inputId, effect.mode, FIELDId);
              if (status != SUCCESS) {
                Logger.error(TAG, `startFieldEffect error number is ${status}`);
                return FAILED;
              } else {
                Logger.info(TAG, `startFieldEffect node successful!!!`);
              }
              break;
            case NodeType.VB:
              let VBId: string = util.generateRandomUUID(true);
              status = audioNapi.multiStartVBEffect(inputId, effect.mode, VBId);
              if (status != SUCCESS) {
                Logger.error(TAG, `startVBEffect error number is ${status}`);
                return FAILED;
              } else {
                Logger.info(TAG, `startVBEffect node successful!!!`);
              }
              break;
            case NodeType.EQ:
              let EQId: string = util.generateRandomUUID(true);
              if (effect.gains == undefined || effect.gains.length == 0) {
                status = audioNapi.multiSetEqualizerMode(effect.mode, EQId, inputId);
              } else {
                status = audioNapi.multiSetEqualizerFrequencyBandGains(effect.gains, EQId, inputId);
              }
              if (status != SUCCESS) {
                Logger.error(TAG, `startVBEffect error number is ${status}`);
                return FAILED;
              } else {
                Logger.info(TAG, `startVBEffect node successful!!!`);
              }
              break;
          }
        }
        Logger.info(TAG, `all effect node of input inited, inputId :${inputId}, pipelineId:${pipelineId}`);
      }
      Logger.info(TAG, `all effect node of pipeline inited, pipelineId:${pipelineId}`);
    }
    return SUCCESS;
  }

  destroyPipeline() {
    Logger.info(TAG, "destroyPipeline start");
    if (preparedPipelineList.length == 0) {
      return;
    }
    preparedPipelineList.forEach((pipelineId: string) => {
      pipelineIdToPipelineInfoMap.get(pipelineId)?.inputIdList.forEach((inputId: string) => {
        audioNapi.multiPipelineEnvPrepare(pipelineId);
        audioNapi.multiDeleteSong(inputId);
      })
    })
    if (preparedPipelineList.length >= 1) {
      audioNapi.destroyMultiPipeline();
    } else {
      Logger.error(TAG, 'pipelineList is blank while destroy');
    }
    preparedPipelineList.length = 0;
    Logger.info(TAG, "destroyPipeline end");
  }

  async saveSingleFile(pipelineId: string, pcmBuffer: ArrayBuffer[], audioFormate: string, outputInfo: OutputFile,
    multiRenderFrameFlag: boolean): Promise<number> {

    pcmBuffer.forEach((value: ArrayBuffer, index) => {
      Logger.info(TAG, `saveSingleFile pcmBuffer of ---${index}---length : ${value.byteLength}`)
    })
    try {
      const now = new Date();
      let newFileName1 = `${outputInfo.nameTemplate}${multiRenderFrameFlag ? "_Voice" :
        ""}_${getDateStringWithTimeStamp(now.getTime(), 1)}.wav`;
      let newFileName2 = `${outputInfo.nameTemplate}${multiRenderFrameFlag ? "_Accompaniment" :
        ""}_${getDateStringWithTimeStamp(now.getTime(), 1)}.wav`;
      const outputFilePath = multiRenderFrameFlag ? [`${outputPath}/${newFileName1}`, `${outputPath}/${newFileName2}`] :
        [`${outputPath}/${newFileName1}`];
      outputFilePath.forEach((path: string, index: number) => {
        let file = fs.openSync(path, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
        Logger.info(TAG, `SaveBuffer filePath : ${outputFilePath}`);
        if (audioFormate === '.wav') {
          // 处理wav
          let wavBuffer =
            writeWavFileHeader(pcmBuffer[index], outputInfo.sampleRate, outputInfo.channels, outputInfo.bitDepth);
          let writeLen = fs.writeSync(file.fd, wavBuffer);
          Logger.info(TAG, `SaveWavBuffer writeLen : ${writeLen}`);
          fs.closeSync(file);
        } else {
          // 处理pcm
          let writeLen = fs.writeSync(file.fd, pcmBuffer[0]);
          Logger.info(TAG, `SavePcmBuffer writeLen : ${writeLen}`);
          fs.closeSync(file);
        }
      })

      outputInfo.outputFileList = outputFilePath;
      const sceneInfoId = pipelineIdToPipelineInfoMap.get(pipelineId)?.sceneInfoId;
      if (sceneInfoId === undefined) {
        return 1;
      }
      let expectedFileMd5: string[] | undefined =
        sceneInfoIdToSceneInfoMap.get(sceneInfoId)?.expectedOutput.md5Value;


      let outputFileMd5: string[] = [];
      try {
        outputFileMd5 = outputFilePath.map((path: string) => {
          return getMD5OfFile(path);
        })
      } catch (e) {
        Logger.error(TAG, `Get MD5 failed, error : ${e}`)
      }
      if (expectedFileMd5 != undefined && checkMd5IsEqual(outputFileMd5, expectedFileMd5)) {
        Logger.error(TAG,
          `file MD5 compare ERROR --- file:${outputFilePath}; md5Value:${outputFileMd5}; expectedMd5:${expectedFileMd5}`);
        this.getUIContext().showAlertDialog({
          message: '输出文件与预期不符失败\n请单击弹窗外任意处关闭弹窗',
          autoCancel: true
        })
      } else {
        Logger.info(TAG,
          `file MD5 compare SUCCESS --- file1:${outputFilePath}; md5Value:${outputFileMd5};`);
      }
      return 0;
    } catch (e) {
      Logger.error(TAG, `SaveBuffer catch: ${JSON.stringify(e)}`)
      return 1;
    }
  }

  async initAutoTestDir() {
    let documentSaveOptions = new picker.DocumentSaveOptions();
    documentSaveOptions.pickerMode = 1;
    let context = getContext() as common.Context;
    let documentPicker = new picker.DocumentViewPicker(context);
    documentPicker.save(documentSaveOptions, (err: BusinessError, documentSaveResult: Array<string>) => {
      if (err) {
        Logger.error(TAG,
          `DocumentViewPicker.save failed with err, code is: ${err.code}, message is: ${err.message}`);
        return;
      }
      Logger.info(TAG,
        'DocumentViewPicker.save successfully, documentSaveResult uri: ' + JSON.stringify(documentSaveResult));
      let uri = documentSaveResult[0];
      let filePath = new fileUri.FileUri(uri + `/init.txt`).path;
      // documentSaveResult数组中只有一个路径
      let file = fs.openSync(filePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
      fs.writeSync(file.fd, new ArrayBuffer(0));
      Logger.info(TAG, `audiodemo dir init success`);
      fs.closeSync(file);
      dirCheckList.forEach((dir: string) => {
        let dirPath = new fileUri.FileUri(`${uri}/${dir}`).path;
        fs.mkdirSync(dirPath);
      })
    })
  }

  makeNodeMapNotBlank(nodeId: string): NodeInfo {
    let nodeInfo = nodeInfoIdToNodeInfoMap.get(nodeId);
    if (nodeInfo == undefined) {
      nodeInfo = { nodeInfoId: "", type: "" }
    }
    return nodeInfo;
  }

  generateSingleReportInfo(pipelineId: string) {
    const now = new Date();
    const endTime = now.getTime();
    const csvFileSavePath = `execute_report_${getDateStringWithTimeStamp(endTime, 1)}.csv`;
    const singleReport = pipelineIdToSingleReportMap.get(pipelineId);
    if (singleReport == undefined) {
      return;
    }
    singleReport.endTimeStamp = endTime;
    singleReport.endTime = getDateStringWithTimeStamp(endTime);
    singleReport.costTime = endTime - singleReport.startTimeStamp;
    singleReport.result = true;
    singleReport.exeReportPath = csvFileSavePath;
    Logger.info(TAG, `singleReport of pipeline:${pipelineId} generated;`);
  }

  async saveSingleCaseExecuteReport(summary: string) {
    Logger.info(TAG, `saveSingleCaseExecuteReport start summary=${summary}`);
    const csvFileSavePath = `${outputPath}/execute_report_${getDateStringWithTimeStamp(new Date().getTime(), 1)}.csv`;
    let csvFileInfoText = await this.getCsvTextOfSingeCase(summary);
    let csvText = `执行用例: ${summary},,,,,,,\n` + csvReportHeaderText + csvFileInfoText + ",,,,,,,\n";
    const csvTextWithBom = this.addBOMToStartOfString(csvText);
    let file = fs.openSync(csvFileSavePath,
      fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
    let writeLen = fs.writeSync(file.fd, csvTextWithBom);
    fs.closeSync(file);
    Logger.info(TAG, `Save CSV file suncess writeLen : ${writeLen}`);
  }

  async saveAllCaseExecuteReport() {
    Logger.info(TAG, "saveAllCaseExecuteReport start");
    const csvFileSavePath = `${outputPath}/execute_report_${getDateStringWithTimeStamp(new Date().getTime(), 1)}.csv`;
    let csvText = ``;
    const summaryArray = Array.from(jsonSummaryToJsonFileInfoMap.keys());
    const promises = summaryArray.map(async (summary: string) => {
      let csvFileInfoText = await this.getCsvTextOfSingeCase(summary);
      csvText += `执行用例: ${summary},,,,,,,\n` + csvReportHeaderText + csvFileInfoText + ",,,,,,,\n";
      return Promise.resolve();
    })
    await Promise.all(promises);
    const csvTextWithBom = this.addBOMToStartOfString(csvText);
    let file = fs.openSync(csvFileSavePath,
      fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
    let writeLen = fs.writeSync(file.fd, csvTextWithBom);
    fs.closeSync(file);
    Logger.info(TAG, `Save CSV file suncess writeLen : ${writeLen}`);
  }

  async getCsvTextOfSingeCase(summary: string): Promise<string> {
    let csvText = ``;
    let texts: string[] = [];
    const promises = jsonSummaryToJsonFileInfoMap.get(summary)?.sceneInfos.map(async (scene) => {
      let pipelineId = sceneInfoIdToSceneInfoMap.get(scene)?.pipelineId;
      if (pipelineId == undefined) {
        return;
      }
      let singleReport = pipelineIdToSingleReportMap.get(pipelineId)
      if (singleReport == undefined) {
        return;
      }
      let text = await this.getCsvTextFromSingleReport(singleReport);
      texts.push(text);
    })
    if (promises == undefined) {
      return "VALID";
    }
    await Promise.all(promises);
    texts.forEach((text) => {
      csvText += `${text}\n`;
    })
    return csvText;
  }

  async getCsvTextFromSingleReport(singleReport: ExecuteReport): Promise<string> {
    let inputFileReportText = ``;
    let outputFileReportText = ``;
    let inputDurationText = ``;
    await this.getMetadataCsvTextOfInputFiles(sceneInfoIdToSceneInfoMap.get(singleReport.sceneInfoId)?.inputFiles)
      .then((inputCsvRet: string[]) => {
        inputFileReportText = inputCsvRet[0];
        inputDurationText = inputCsvRet[1];
      })
    await this.getMetadataCsvTextOfoutputFiles(sceneInfoIdToSceneInfoMap.get(singleReport.sceneInfoId)?.outputFile.outputFileList)
      .then((outputCsvRet: string) => {
        outputFileReportText = outputCsvRet;
      })

    let ret: string =
      `"${singleReport.sceneName}","${singleReport.startTime}","${singleReport.endTime}","${singleReport.costTime /
        1000} s","${inputFileReportText}","${inputDurationText}","${outputFileReportText}","${singleReport.result ?
        "执行成功" : "执行异常"}"`;
    return ret;
  }

  async getMetadataCsvTextOfInputFiles(inputFiles: InputFile[] | undefined): Promise<string[]> {
    if (inputFiles == undefined) {
      return ["InputFiles open FAILED", ""];
    }
    let ret: string = ``;
    let durationStr: string = ``;
    const promises = inputFiles.map(async (input) => {
      Decimal.set({
        precision: 4,
      });
      let inputFilePath = `${inputDir}/${input.path}`;
      let file = fs.openSync(inputFilePath, fs.OpenMode.READ_WRITE);
      let fsStat = fs.statSync(inputFilePath);
      let buffer = new ArrayBuffer(fsStat.size);
      fs.readSync(file.fd, buffer);
      const wavView = new Uint8Array(buffer);
      let wavHeaderBuffer = wavView.slice(0, 44).buffer;
      const dataView = new DataView(wavHeaderBuffer);
      // 获取通道数, 通道数位于22字节
      const channels: number = dataView.getUint16(22, true);
      // 获取采样率, 采样率位于第 24 到28 字节
      const sampleRate: number = dataView.getUint32(24, true);
      // 获取位深，位深位于 34字节
      const bitsPerSample: number = dataView.getUint16(34, true);
      return media.createAVMetadataExtractor().then(async (avMetadataExtractor) => {
        // 设置fdSrc
        avMetadataExtractor.fdSrc = fs.openSync(inputFilePath);
        // 获取元数据（promise模式）
        let metadata = await avMetadataExtractor.fetchMetadata();
        let fileLen = new Decimal(fsStat.size / 1024 / 1024 * 100).ceil();
        console.info(TAG, `get meta data, media length: ${Number(metadata.duration) / 1000} s`);
        ret += `--------------------------\r
        输入文件路径: ${inputFilePath}\r
        文件大小: ${(Number(fileLen) / 100).toString()} MB\r
        音频长度: ${Number(metadata.duration) / 1000} s\r
        音频格式(采样率/声道数/位深): ${sampleRate}/${channels}/${bitsPerSample}\r
        `
        durationStr += `--------------------------\r
      ${Number(metadata.duration) / 1000} s\r
      `
      })
    })
    await Promise.all(promises);
    return [ret, durationStr];
  }

  async getMetadataCsvTextOfoutputFiles(outputFiles: string[] | undefined): Promise<string> {
    if (outputFiles == undefined) {
      return "OutputFiles open FAILED";
    }
    let ret: string = ``;
    for (const outputPath of outputFiles) {
      Decimal.set({
        precision: 4,
      });
      let file = fs.openSync(outputPath, fs.OpenMode.READ_WRITE);
      let fsStat = fs.statSync(outputPath);
      let buffer = new ArrayBuffer(fsStat.size);
      fs.readSync(file.fd, buffer);
      const wavView = new Uint8Array(buffer);
      let wavHeaderBuffer = wavView.slice(0, 44).buffer;
      const dataView = new DataView(wavHeaderBuffer);
      // 获取通道数, 通道数位于22字节
      const channels: number = dataView.getUint16(22, true);
      // 获取采样率, 采样率位于第 24 到28 字节
      const sampleRate: number = dataView.getUint32(24, true);
      // 获取位深，位深位于 34字节
      const bitsPerSample: number = dataView.getUint16(34, true);

      let avMetadataExtractor = await media.createAVMetadataExtractor();
      // 设置fdSrc
      avMetadataExtractor.fdSrc = fs.openSync(outputPath);
      // 获取元数据（promise模式）
      let metadata = await avMetadataExtractor.fetchMetadata();
      let fileLen = new Decimal(fsStat.size / 1024 / 1024 * 100).ceil();
      console.info(TAG, `get meta data, media length: ${Number(metadata.duration) / 1000} s`);
      ret += `--------------------------\r
      输出文件路径: ${outputPath}\r
      文件大小: ${(Number(fileLen) / 100).toString()} MB\r
      音频长度: ${Number(metadata.duration) / 1000} s\r
      音频格式(采样率/声道数/位深): ${sampleRate}/${channels}/${bitsPerSample}\r
      `
    }
    return new Promise((resolve: Function) => {
      resolve(ret);
    });
  }

  addBOMToStartOfString(originStr:string):string {
    const encoder = util.TextEncoder.create();
    const bom = encoder.encode('\uFEFF');
    const content = encoder.encode(originStr);
    const resultArray = new Uint8Array(bom.length + content.length);
    resultArray.set(bom, 0);
    resultArray.set(content, bom.length);
    const decoder = util.TextDecoder.create();
    return decoder.decode(resultArray);
  }
}