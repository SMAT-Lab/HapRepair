/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

import audioNapi from 'libentry.so';
import { TextModifier } from '@kit.ArkUI';
import { OutputFile } from './autoTest/Interface';
import { Logger } from '../utils/Logger';
const TAG: string = 'AudioEditTestApp_AudioEdit';


@Concurrent
export async function saveFileBuffer(): Promise<ArrayBuffer> {
    let pcmBuffer = audioNapi.saveFileBuffer();
    return pcmBuffer;
}

@Concurrent
export async function multiPipelineSaveFileBuffer(pipelineId: string, outputInfo:OutputFile, multiRenderFlag:boolean): Promise<ArrayBuffer[]> {
    const TAG: string = 'AudioEditTestApp_AudioEdit';
    Logger.info(TAG, `multiPipelineSaveFileBuffer start---pipelineId:${pipelineId}, multiRenderFlag:${multiRenderFlag}`)
    audioNapi.multiPipelineEnvPrepare(pipelineId);
    audioNapi.multiSetFormat(outputInfo?.channels, outputInfo?.sampleRate, outputInfo?.bitDepth);
    let pcmBuffer = audioNapi.multiSaveFileBuffer();

    let tapBuffer: ArrayBuffer = new ArrayBuffer(0);
    if (multiRenderFlag){
        tapBuffer = audioNapi.multiGetSecondOutputAudio();
    }
    return [pcmBuffer as ArrayBuffer, tapBuffer];
}

@Concurrent
export async function RealTimeSaveFileBuffer(): Promise<ArrayBuffer> {
    let pcmBuffer = audioNapi.realTimeSaveFileBuffer();
    return pcmBuffer;
}

export class MainTitleTextModifier extends TextModifier {
    userStyle: boolean = true;

    applyNormalAttribute(instance: TextAttribute): void {
        if (this.userStyle) {
            instance.fontColor($r('app.color.index_title_text_color'));
        }
    }
}


export function writeString(dv: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
        dv.setUint8(offset + i, str.charCodeAt(i));
    }
}

export function concatArrayBuffer(wavHeaderBuffer: ArrayBuffer, pcmBuffer: ArrayBuffer) {
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

export function writeWavFileHeader(pcmBuffer: ArrayBuffer, sampleRate: number, channels: number, bitsPerSample: number): ArrayBuffer {
    Logger.info(TAG,
        `writeWavFileHeader sampleRate : ${sampleRate} --- channels : ${channels} --- bitsPerSample : ${bitsPerSample}`);
    const header = new ArrayBuffer(44);
    const dv = new DataView(header);
    // 写入RIFF快
    writeString(dv, 0, 'RIFF');
    let totalDataLen = pcmBuffer.byteLength + 44;
    dv.setUint32(4, totalDataLen, true);
    writeString(dv, 8, 'WAVE');
    // 写入 fmt 快
    writeString(dv, 12, 'fmt ');
    // fmt快大小
    dv.setUint32(16, 16, true);
    // 格式类别（1 - PCM， 3 - IEEE 浮点数）
    dv.setUint16(20, 3, true);
    if (bitsPerSample === 32) {
        dv.setUint16(20, 3, true);
    } else {
        dv.setUint16(20, 1, true);
    }
    // 声道数
    dv.setUint16(22, channels, true);
    // 采样率
    dv.setUint32(24, sampleRate, true);
    // 比特率
    let byteRate = bitsPerSample * sampleRate * channels / 8;
    dv.setUint32(28, byteRate, true);
    // 每个采样点的字节数
    dv.setUint16(32, channels * bitsPerSample / 8, true);
    // 位深
    dv.setUint16(34, bitsPerSample, true);
    // 写入data 快
    writeString(dv, 36, 'data');
    // 数据快大小
    dv.setUint32(40, pcmBuffer.byteLength, true);

    let resultBuffer = concatArrayBuffer(header, pcmBuffer);
    return resultBuffer;
}