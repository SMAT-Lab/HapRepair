/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef AUDIOEDITTESTAPP_REALTIMEPLAYING_H
#define AUDIOEDITTESTAPP_REALTIMEPLAYING_H

#include "ohaudio/native_audio_suite_base.h"
#include "ohaudio/native_audiostream_base.h"

extern OH_AudioRenderer *audioRenderer;

extern OH_AudioStreamBuilder *rendererBuilder;

// 实时播放，一次渲染是否完成
extern bool g_playFinishedFlag;

extern char *g_playAudioData;

extern int32_t g_playDataSize;

// 是否录制
extern bool g_isRecord;

// 实时播放，用于保存音频数据，具体大小根据需要保存的文件大小而变化
extern char *g_playTotalAudioData;

// 实时播放需要保存的音频总大小
extern int32_t g_playResultTotalSize;

OH_AudioSuite_Result ProcessPipeline();

OH_AudioSuite_Result OneRenDerFrame(int32_t audioDataSize, int32_t *writeSize);

OH_AudioData_Callback_Result PlayAudioRendererOnWriteData(OH_AudioRenderer *renderer,
    void *userData, void *audioData, int32_t audioDataSize);

void ReleaseExistingResources();

#endif //AUDIOEDITTESTAPP_REALTIMEPLAYING_H