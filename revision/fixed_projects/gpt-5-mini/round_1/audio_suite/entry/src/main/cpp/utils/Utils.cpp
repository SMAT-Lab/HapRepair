/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "Utils.h"
#include "audioSuiteError/AudioSuiteError.h"
#include <hilog/log.h>

const int CONSTANT_0 = 0;

const int CONSTANT_1 = 1;
const int GLOBAL_RESMGR = 0xFF00;
const char *UTILS_TAG = "[AudioEditTestApp_utils_cpp]";

// 解析 napi 字符串参数
napi_status ParseNapiString(napi_env env, napi_value value, std::string &result)
{
    size_t size;
    napi_status status = napi_get_value_string_utf8(env, value, nullptr, CONSTANT_0, &size);
    if (status != napi_ok) {
        return status;
    }

    result.resize(size + CONSTANT_1); // 包含结尾的空字符
    status = napi_get_value_string_utf8(env, value, const_cast<char *>(result.data()), size + CONSTANT_1, nullptr);

    return status;
}

void GetBitsPerSampleAndStreamFormat(const OH_AudioFormat& g_audioFormatOutput,
    int32_t* bitsPerSample, OH_AudioStream_SampleFormat* streamSampleFormat)
{
    if (g_audioFormatOutput.sampleFormat == OH_Audio_SampleFormat::AUDIO_SAMPLE_U8) {
        *bitsPerSample = DemoBitsPerSample::DEMO_BITSPERSAMPLE_8;
        *streamSampleFormat = OH_AudioStream_SampleFormat::AUDIOSTREAM_SAMPLE_U8;
    } else if (g_audioFormatOutput.sampleFormat == OH_Audio_SampleFormat::AUDIO_SAMPLE_S16LE) {
        *bitsPerSample = DemoBitsPerSample::DEMO_BITSPERSAMPLE_16;
        *streamSampleFormat = OH_AudioStream_SampleFormat::AUDIOSTREAM_SAMPLE_S16LE;
    } else if (g_audioFormatOutput.sampleFormat == OH_Audio_SampleFormat::AUDIO_SAMPLE_S24LE) {
        *bitsPerSample = DemoBitsPerSample::DEMO_BITSPERSAMPLE_24;
        *streamSampleFormat = OH_AudioStream_SampleFormat::AUDIOSTREAM_SAMPLE_S24LE;
    }  else {
        *bitsPerSample = DemoBitsPerSample::DEMO_BITSPERSAMPLE_32;
        *streamSampleFormat = OH_AudioStream_SampleFormat::AUDIOSTREAM_SAMPLE_F32LE;
    }
}

// 设置采样率
OH_Audio_SampleRate SetSamplingRate(int32_t sampleRate)
{
    switch (sampleRate) {
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_8000):
            return OH_Audio_SampleRate::SAMPLE_RATE_8000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_11025):
            return OH_Audio_SampleRate::SAMPLE_RATE_11025;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_12000):
            return OH_Audio_SampleRate::SAMPLE_RATE_12000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_16000):
            return OH_Audio_SampleRate::SAMPLE_RATE_16000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_22050):
            return OH_Audio_SampleRate::SAMPLE_RATE_22050;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_24000):
            return OH_Audio_SampleRate::SAMPLE_RATE_24000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_32000):
            return OH_Audio_SampleRate::SAMPLE_RATE_32000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_44100):
            return OH_Audio_SampleRate::SAMPLE_RATE_44100;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_48000):
            return OH_Audio_SampleRate::SAMPLE_RATE_48000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_64000):
            return OH_Audio_SampleRate::SAMPLE_RATE_64000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_88200):
            return OH_Audio_SampleRate::SAMPLE_RATE_88200;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_96000):
            return OH_Audio_SampleRate::SAMPLE_RATE_96000;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_176400):
            return OH_Audio_SampleRate::SAMPLE_RATE_176400;
        case static_cast<int32_t>(OH_Audio_SampleRate::SAMPLE_RATE_192000):
            return OH_Audio_SampleRate::SAMPLE_RATE_192000;
        default:
            return OH_Audio_SampleRate::SAMPLE_RATE_48000;
    }
}

// 设置声道
OH_AudioChannelLayout SetChannelLayout(int32_t channels)
{
    OH_AudioChannelLayout audioChannelLayout;
    switch (channels) {
        case DemoChannels::DEMO_CHANNELS_1:
            audioChannelLayout = CH_LAYOUT_MONO;
            break;
        case DemoChannels::DEMO_CHANNELS_2:
            audioChannelLayout = CH_LAYOUT_STEREO;
            break;
        default:
            audioChannelLayout = CH_LAYOUT_STEREO_DOWNMIX;
            break;
    }
    return audioChannelLayout;
}

// 设置位深
OH_Audio_SampleFormat SetSampleFormat(int32_t bitsPerSample)
{
    OH_Audio_SampleFormat audioSampleFormat;
    switch (bitsPerSample) {
        case DemoBitsPerSample::DEMO_SAMPLE_U8:
            audioSampleFormat = OH_Audio_SampleFormat::AUDIO_SAMPLE_U8;
            break;
        case DemoBitsPerSample::DEMO_SAMPLE_S16LE:
            audioSampleFormat = OH_Audio_SampleFormat::AUDIO_SAMPLE_S16LE;
            break;
        case DemoBitsPerSample::DEMO_SAMPLE_S246E:
            audioSampleFormat = OH_Audio_SampleFormat::AUDIO_SAMPLE_S24LE;
            break;
        case DemoBitsPerSample::DEMO_SAMPLE_S32LE:
            audioSampleFormat = OH_Audio_SampleFormat::AUDIO_SAMPLE_S32LE;
            break;
        case DemoBitsPerSample::DEMO_SAMPLE_F32LE:
            audioSampleFormat = OH_Audio_SampleFormat::AUDIO_SAMPLE_F32LE;
            break;
        default:
            break;
    }
    return audioSampleFormat;
}

// 位深转换
void ConvertBitsPerSample(unsigned int& bitsPerSample)
{
    switch (bitsPerSample) {
        case DemoBitsPerSample::DEMO_BITSPERSAMPLE_8:
            bitsPerSample = DemoBitsPerSample::DEMO_SAMPLE_U8;
            break;
        case DemoBitsPerSample::DEMO_BITSPERSAMPLE_16:
            bitsPerSample = DemoBitsPerSample::DEMO_SAMPLE_S16LE;
            break;
        case DemoBitsPerSample::DEMO_BITSPERSAMPLE_24:
            bitsPerSample = DemoBitsPerSample::DEMO_SAMPLE_S246E;
            break;
        case DemoBitsPerSample::DEMO_BITSPERSAMPLE_32:
            bitsPerSample = DemoBitsPerSample::DEMO_SAMPLE_F32LE;
            break;
        default:
            break;
    }
}

int32_t GetBitsPerSample(OH_Audio_SampleFormat sampleFormat)
{
    switch (sampleFormat) {
        case OH_Audio_SampleFormat::AUDIO_SAMPLE_U8:
            return DemoBitsPerSample::DEMO_BITSPERSAMPLE_8;
        case OH_Audio_SampleFormat::AUDIO_SAMPLE_S16LE:
            return DemoBitsPerSample::DEMO_BITSPERSAMPLE_16;
        case OH_Audio_SampleFormat::AUDIO_SAMPLE_S24LE:
            return DemoBitsPerSample::DEMO_BITSPERSAMPLE_24;
        default:
            return DemoBitsPerSample::DEMO_BITSPERSAMPLE_32;
    }
}

OH_EnvironmentType GetEnvEnumByNumber(int num)
{
    OH_EnvironmentType type;
    switch (num) {
        case ARG_1:
            type = ENVIRONMENT_TYPE_BROADCAST;
            break;
        case ARG_2:
            type = ENVIRONMENT_TYPE_EARPIECE;
            break;
        case ARG_3:
            type = ENVIRONMENT_TYPE_UNDERWATER;
            break;
        case ARG_4:
            type = ENVIRONMENT_TYPE_GRAMOPHONE;
            break;
        default:
            break;
    }
    return type;
}

napi_value ReturnResult(napi_env env, AudioSuiteResult result)
{
    std::string resultMessage = GetErrorMessage(result);
    napi_value sum;
    if (result != AudioSuiteResult::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, UTILS_TAG,
            "result: %{public}d, resultMessage: %{public}s", result, resultMessage.c_str());
    } else {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, UTILS_TAG,
            "result: %{public}d, resultMessage: %{public}s", result, resultMessage.c_str());
    }
    napi_create_int64(env, static_cast<int>(result), &sum);
    return sum;
}

void FreeBuffer(void *buffer)
{
    if (buffer != nullptr) {
        free(buffer);
        buffer = nullptr;
    }
}