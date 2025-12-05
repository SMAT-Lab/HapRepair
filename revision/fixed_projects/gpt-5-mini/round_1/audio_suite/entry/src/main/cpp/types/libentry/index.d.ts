/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

export * from "./multiPipelineIndex";
export const record: () => void;
export const audioRendererInit: () => void;
export const audioRendererDestory: () => void;
export const audioRendererStart: () => void;
export const audioRendererPause: () => void;
export const audioRendererStop: () => void;
export const getRendererState: () => number;
export const registerFinishedCallback: (callBackFull: Function) => boolean;
export const resetTotalWriteAudioDataSize: () => void;
export const realTimeSaveFileBuffer: () => ArrayBuffer;
export const audioEditNodeInit: (workMode: number) => number;
export const audioInAndOutInit: (inputId: string, outputId: string, mixerId: string, fd: number, bufferLength: number) => number;
export const audioEditDestory: () => number;
export const setFormat: (channels: number, sampleRate: number, bitsPerSample: number) => number;
export const setEquailizerMode: (equailizerMode: number, nodeId: string, inputId:string) => number;
export const setEqualizerFrequencyBandGains: (equailizerBanGains: Array<number>, nodeId: string, inputId:string, selectedNodeId?: string) => number;
export const saveFileBuffer: () => ArrayBuffer;
export const startFieldEffect: (inputId: string, mode: number, fieldEffectId: string, selectedNodeId?: string) => number;
export const resetFieldEffect: (inputId: string, mode: number, fieldEffectId: string) => number;
export const startEnvEffect: (inputId: string, envEffectId: string, mode: number, selectedNodeId?: string) => number;
export const resetEnvEffect: (inputId: string, envEffectId: string, mode: number) => number;
export const addAudioSeparation: (mode: number, uuid: string, inputId:string, selectedNodeId?: string) => number;
export const deleteAudioSeparation: (uuid: string) => number;
export const getAudioOfTap:() => ArrayBuffer;
export const addNoiseReduction: (uuid: string, inputId:string, selectedNode?: string) => number;
export const deleteNoiseReduction: (uuid: string) => number;
export const deleteSong: (inputId: string) => number;
export const startVBEffect: (inputId: string, mode: number, voiceBeautifierId: string, selectedNode?: string) => number;
export const resetVBEffect: (inputId: string, mode: number, voiceBeautifierId: string) => number;
export const deleteNode: (nodeId: string) => number;
export const registerAudioFormatCallback: (callBackFull: Function) => string[];
export const getOptions: (nodeId: string) => string;
export const getEffectNodeList: () => Array<number>;