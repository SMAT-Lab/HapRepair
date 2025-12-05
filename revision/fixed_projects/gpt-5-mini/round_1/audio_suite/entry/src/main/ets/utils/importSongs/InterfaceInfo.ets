/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

export interface SongInfo {
    uri: string;
    songName?: string | Resource;
    songType?: string;
    nodes?: Array<Node>;
    nodeId?: string;
    duration?: number;
    singerName?: string | Resource;
    albumCover?: PixelMap;
    albumName?: string | Resource;
    wavBuffer?: ArrayBuffer;
    channels?: number;
    sampleRate?: number;
    bitsPerSample?: number;
    isPlay?: boolean;
    fd?: number;
}

export interface Node {
    id: string;
    type: NodeType;
    color?: BackColor;
    balanceBandGains?: number[];
    soundFiledType?: SoundFiledType;
    environmentType?: EnvironmentType;
    voiceBeautifierType?: VoiceBeautifierType;
    audioSeparationType?: AudioSeparationType;
}

export enum BackColor {
    FIELD_COLOR = '#ffee0b80',
    SEPARATION_COLOR = '#ffae05f6',
    NR_COLOR = '#ff0b71ee',
    EQ_COLOR = '#ffeea20b',
    VB_COLOR = '#ff0f740b',
    MIXER_COLOR = '#ff0d27cf',
    INPUT_COLOR = '#ff1eac35',
    OUTPUT_COLOR = '#ff20c61a',
    ENV_COLOR = '#ffe2ad79'
}

export enum NodeType {
    'EQ' = '均衡',
    'FIELD' = '声场',
    'ENV' = '环境',
    'SEPARATION' = '分离',
    'VOICE' = '人声',
    'ACCOMPANIMENT' = '伴奏',
    'NR' = '降噪',
    'INPUT' = '输入',
    'OUTPUT' = '输出',
    'MIXER' = '混音',
    'VB' = '美化'
}

export const ColorMap = new Map<NodeType, BackColor>([
    [NodeType.EQ, BackColor.EQ_COLOR],
    [NodeType.FIELD, BackColor.FIELD_COLOR],
    [NodeType.VOICE, BackColor.SEPARATION_COLOR],
    [NodeType.ACCOMPANIMENT, BackColor.SEPARATION_COLOR],
    [NodeType.NR, BackColor.NR_COLOR],
    [NodeType.VB, BackColor.VB_COLOR],
    [NodeType.MIXER, BackColor.MIXER_COLOR],
    [NodeType.INPUT, BackColor.INPUT_COLOR],
    [NodeType.OUTPUT, BackColor.OUTPUT_COLOR],
    [NodeType.ENV, BackColor.ENV_COLOR],
    [NodeType.SEPARATION, BackColor.SEPARATION_COLOR]
])

export enum SoundFiledType {
    /**
     * 无
     */
    SOUND_FIELD_CLOSE = 1,
    /**
     * 前置
     */
    SOUND_FIELD_FRONT_FACING = 2,
    /**
     * 宏大
     */
    SOUND_FIELD_GRAND = 3,
    /**
     * 聆听
     */
    SOUND_FIELD_NEAR = 4,
    /**
     * 宽广
     */
    SOUND_FIELD_WIDE = 5,
}

export const SoundFiledMap = new Map<SoundFiledType | undefined, string>([
    [SoundFiledType.SOUND_FIELD_CLOSE, '无'],
    [SoundFiledType.SOUND_FIELD_FRONT_FACING, '前置'],
    [SoundFiledType.SOUND_FIELD_GRAND, '宏大'],
    [SoundFiledType.SOUND_FIELD_NEAR, '聆听'],
    [SoundFiledType.SOUND_FIELD_WIDE, '宽广']
])

export enum EnvironmentType {
    /**
     * Close environment effect type.
     * 
     * @since 22
     */
    ENVIRONMENT_TYPE_CLOSE = 1,
    /**
     * Broadcast environment effect type.
     * 
     * @since 22
     */
    ENVIRONMENT_TYPE_BROADCAST = 2,
    /**
     * Earpiece environment effect type.
     * 
     * @since 22
     */
    ENVIRONMENT_TYPE_EARPIECE = 3,
    /**
     * Earpiece environment effect type.
     * 
     * @since 22
     */
    ENVIRONMENT_TYPE_UNDERWATER = 4,
    /**
     * Gramophone environment effect type.
     * 
     * @since 22
     */
    ENVIRONMENT_TYPE_GRAMOPHONE = 5,
}

export const EnvironmentMap = new Map<EnvironmentType | undefined, string>([
    [EnvironmentType.ENVIRONMENT_TYPE_CLOSE, '无'],
    [EnvironmentType.ENVIRONMENT_TYPE_BROADCAST, '广播'],
    [EnvironmentType.ENVIRONMENT_TYPE_EARPIECE, '听筒'],
    [EnvironmentType.ENVIRONMENT_TYPE_UNDERWATER, '水下'],
    [EnvironmentType.ENVIRONMENT_TYPE_GRAMOPHONE, '留声机']
])

export enum VoiceBeautifierType {
    /**
     * Normal voice beautifier type.
     * 
     * @since 22
     */
    VOICE_BEAUTIFIER_TYPE_NORMAL,
    /**
     * Clear voice beautifier type.
     * 
     * @since 22
     */
    VOICE_BEAUTIFIER_TYPE_CLEAR,
    /**
     * Theatre voice beautifier type.
     * 
     * @since 22
     */
    VOICE_BEAUTIFIER_TYPE_THEATRE,
    /**
     * CD voice beautifier type.
     * 
     * @since 22
     */
    VOICE_BEAUTIFIER_TYPE_CD,
    /**
     * Studio voice beautifier type.
     * 
     * @since 22
     */
    VOICE_BEAUTIFIER_TYPE_STUDIO,
}

export const VoiceBeautifierMap = new Map<VoiceBeautifierType | undefined, string>([
    [VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_NORMAL, '普通'],
    [VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_CLEAR, '清澈'],
    [VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_THEATRE, '剧场'],
    [VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_CD, 'CD音效'],
    [VoiceBeautifierType.VOICE_BEAUTIFIER_TYPE_STUDIO, '录音棚'],
])

export enum AudioSeparationType {
    /**
     * @background sound type.
     * 
     * @since 22
     */
    AUDIO_NODE_BACKGROUND_SOUND_OUTPORT_TYPE = 0,
    /**
     * @human sound type.
     * 
     * @since 22
     */
    AUDIO_NODE_HUMAN_SOUND_OUTPORT_TYPE = 1,
}

export const AudioSeparationMap = new Map<AudioSeparationType | undefined, string>([
    [AudioSeparationType.AUDIO_NODE_HUMAN_SOUND_OUTPORT_TYPE, '人声'],
    [AudioSeparationType.AUDIO_NODE_BACKGROUND_SOUND_OUTPORT_TYPE, '伴奏']
])

// 定义映射表：数字 -> NodeType
export const NumberToNodeTypeMap = new Map<number, NodeType>([
  [201, NodeType.EQ],
  [202, NodeType.NR],
  [203, NodeType.FIELD],
  [204, NodeType.SEPARATION],
  [205, NodeType.VB],
  [206, NodeType.ENV]
]);