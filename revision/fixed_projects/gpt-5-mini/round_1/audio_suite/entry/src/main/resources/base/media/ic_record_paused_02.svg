/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef AUDIOEDITTESTAPP_UTILS_H
#define AUDIOEDITTESTAPP_UTILS_H

#include <string>
#include "audioSuiteError/AudioSuiteError.h"
#include "napi/native_api.h"
#include "ohaudio/native_audio_suite_base.h"
#include "ohaudio/native_audiostream_base.h"

enum {
    ARG_0 = 0,
    ARG_1 = 1,
    ARG_2 = 2,
    ARG_3 = 3,
    ARG_4 = 4,
    ARG_5 = 5,
    ARG_6 = 6,
    ARG_7 = 7,
    ARG_8 = 8,
};

enum DemoBitsPerSample {
    DEMO_SAMPLE_U8 = 0,
    DEMO_SAMPLE_S16LE = 1,
    DEMO_SAMPLE_S246E = 2,
    DEMO_SAMPLE_S32LE = 3,
    DEMO_SAMPLE_F32LE = 4,

    DEMO_BITSPERSAMPLE_8 = 8,
    DEMO_BITSPERSAMPLE_16 = 16,
    DEMO_BITSPERSAMPLE_24 = 24,
    DEMO_BITSPERSAMPLE_32 = 32,
};

enum DemoChannels {
    DEMO_CHANNELS_1 = 1,
    DEMO_CHANNELS_2 = 2,
};

struct FieldEffectParams {
    std::string inputId;
    unsigned int mode = 0;
    std::string fieldEffectId;
    std::string selectedNodeId;
};

struct FieldEffectParams {
    std::string inputId;
    unsigned int mode = 0;
    std::string fieldEffectId;
    std::string selectedNodeId;
};

napi_status ParseNapiString(napi_env env, napi_value value, std::string &result);

void GetBitsPerSampleAndStreamFormat(const OH_AudioFormat& g_audioFormatOutput,
    int32_t* bitsPerSample, OH_AudioStream_SampleFormat* streamSampleFormat);

// 设置采样率
OH_Audio_SampleRate SetSamplingRate(int32_t sampleRate);

// 设置声道
OH_AudioChannelLayout SetChannelLayout(int32_t channels);

// 设置位深
OH_Audio_SampleFormat SetSampleFormat(int32_t bitsPerSample);

// 位深转换
void ConvertBitsPerSample(unsigned int& bitsPerSample);

int32_t GetBitsPerSample(OH_Audio_SampleFormat sampleFormat);

OH_EnvironmentType GetEnvEnumByNumber(int num);

napi_value ReturnResult(napi_env env, AudioSuiteResult result);

void freeBuffer(void *buffer);
#endif //#define AUDIOEDITTESTAPP_UTILS_H
