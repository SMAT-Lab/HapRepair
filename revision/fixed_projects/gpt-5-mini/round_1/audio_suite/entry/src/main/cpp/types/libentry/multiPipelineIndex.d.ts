/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

export const audioEditNodeInitMultiPipeline: (inputId: string) => number;
export const multiAudioInAndOutInit: (inputId: string, outputId: string, mixerId: string, fd: number, bufferLength: number) => number;
export const multiPipelineEnvPrepare: (pipelineId: string) => number;
export const multiSetFormat: (channels: number, sampleRate: number, bitsPerSample: number) => number;
export const multiSetEqualizerMode: (equalizerMode: number, nodeId: string, inputId:string) => number;
export const multiSetEqualizerFrequencyBandGains: (equalizerBanGains: Array<number>, nodeId: string, inputId:string, selectedNodeId?: string) => number;
export const multiStartFieldEffect: (inputId: string, mode: number, fieldEffectId: string, selectedNodeId?: string) => number;
export const multiStartEnvEffect: (inputId: string, envEffectId: string, mode: number, selectedNodeId?: string) => number;
export const multiAddAudioSeparation: (mode: number, uuid: string, inputId:string, selectedNodeId?: string) => number;
export const multiAddNoiseReduction: (uuid: string, inputId:string, selectedNode?: string) => number;
export const multiStartVBEffect: (inputId: string, mode: number, voiceBeautifierId: string, selectedNode?: string) => number;
export const multiSaveFileBuffer: () => ArrayBuffer;
export const multiGetSecondOutputAudio:() => ArrayBuffer;
export const multiDeleteSong: (inputId: string) => number;
export const destroyMultiPipeline: () => number;