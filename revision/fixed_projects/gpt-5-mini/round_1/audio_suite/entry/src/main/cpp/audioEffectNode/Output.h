/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef AUDIOEDITTESTAPP_OUTPUT_H
#define AUDIOEDITTESTAPP_OUTPUT_H

#include <string>
#include "ohaudio/native_audio_suite_base.h"

struct AudioRenderContext {
    char *totalAudioData;
    char *tapTotalAudioData;
    int32_t frameSize;
    bool &finishedFlag;
    int32_t resultTotalSize;
    int32_t tapResultTotalSize;
};

extern OH_AudioFormat g_audioFormatOutput;

extern bool g_multiRenderFrameFlag;

extern bool g_globalFinishFlag;

extern char *g_tapTotalBuff;

extern int32_t g_tapDataTotalSize;

OH_AudioSuite_Result RenDerFrame();

OH_AudioSuite_Result StartPipelineAndCheckState();

OH_AudioSuite_Result AudioRenderFrame(char *totalAudioData, char *tapTotalAudioData,
    int32_t frameSize, bool &finishedFlag);

void SaveBuffer(char *totalData, int32_t &totalSize, void *buffer, int32_t bufferSize);

void LogRenderResult(OH_AudioSuite_Result result, int32_t requestFrameSize, int32_t writeSize,
    bool finishedFlag, std::string logType);

void UpdateGlobalBuffers(AudioRenderContext &context);

#endif //AUDIOEDITTESTAPP_OUTPUT_H