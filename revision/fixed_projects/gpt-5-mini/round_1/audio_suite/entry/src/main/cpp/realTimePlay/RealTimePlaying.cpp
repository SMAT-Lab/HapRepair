/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "RealTimePlaying.h"
#include "hilog/log.h"
#include "../audioEffectNode/Input.h"
#include "ohaudio/native_audio_suite_engine.h"
#include "ohaudio/native_audiorenderer.h"
#include "ohaudio/native_audiostreambuilder.h"
#include "./callback/RegisterCallback.h"
#include "../audioEffectNode/Input.h"

const int GLOBAL_RESMGR = 0xFF00;
const char *REAL_TIME_PLAYING_TAG = "[AudioEditTestApp_RealTimePlaying_cpp]";

const int CONSTANT_0 = 0;

OH_AudioRenderer *audioRenderer = nullptr;

OH_AudioStreamBuilder *rendererBuilder = nullptr;

bool g_playFinishedFlag = false;

char *g_playAudioData = (char *)malloc(g_playDataSize * 5);

int32_t g_playDataSize = 0;

bool g_isRecord = false;

char *g_playTotalAudioData = (char *)malloc(1024 * 1024 * 100);

int32_t g_playResultTotalSize = 0;

OH_AudioSuite_Result ProcessPipeline()
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG, "audioEditTest ProcessPipeline start");
    // 获取管线状态
    OH_AudioSuite_PipelineState pipeLineState;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_GetPipelineState(g_audioSuitePipeline, &pipeLineState);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
        "audioEditTest OH_AudioSuiteEngine_GetPipelineState result: %{public}d --- pipeLineState: %{public}d",
        static_cast<int>(result), static_cast<int>(pipeLineState));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }

    // 启动管线
    if (pipeLineState != OH_AudioSuite_PipelineState::AUDIOSUITE_PIPELINE_RUNNING) {
        result = OH_AudioSuiteEngine_StartPipeline(g_audioSuitePipeline);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
            "audioEditTest OH_AudioSuiteEngine_StartPipeline result: %{public}d", static_cast<int>(result));
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
    }
    return result;
}

OH_AudioSuite_Result OneRenDerFrame(int32_t audioDataSize, int32_t *writeSize)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG, "audioEditTest OneRenDerFrame start");
    ProcessPipeline();
    if (audioDataSize <= CONSTANT_0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
            "audioEditTest OH_AudioSuiteEngine_RenderFrame audioDataSize is %{public}d",
            static_cast<int>(audioDataSize));
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_SYSTEM;
    }
    char *audioData = (char *)malloc(audioDataSize);
    OH_AudioSuite_Result result =
        OH_AudioSuiteEngine_RenderFrame(g_audioSuitePipeline, audioData,
                                        audioDataSize, writeSize, &g_playFinishedFlag);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
        "audioEditTest OH_AudioSuiteEngine_RenderFrame audioDataSize: %{public}d, writeSize:%{public}d "
        "g_playFinishedFlag : %{public}s, result: %{public}d",
        audioDataSize, *writeSize, (g_playFinishedFlag ? "true" : "false"), static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
            "audioEditTest OH_AudioSuiteEngine_RenderFrame result is %{public}d", static_cast<int>(result));
    }
    // 每次保存一次获取的buffer值
    g_playAudioData = (char *)malloc(*writeSize);
    std::copy(audioData, audioData + *writeSize, static_cast<char *>(g_playAudioData));
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
        "audioEditTest OH_AudioSuiteEngine_RenderFrame writeSize : %{public}d, g_playFinishedFlag: %{public}s",
        *writeSize, (g_playFinishedFlag ? "true" : "false"));
    return result;
}

OH_AudioData_Callback_Result PlayAudioRendererOnWriteData(OH_AudioRenderer *renderer,
    void *userData, void *audioData, int32_t audioDataSize)
{
    if (renderer == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
            "audioEditTest PlayAudioRendererOnWriteData renderer is nullptr");
        return AUDIO_DATA_CALLBACK_RESULT_INVALID;
    }
    if (audioData == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
            "audioEditTest PlayAudioRendererOnWriteData audioData is nullptr");
        return AUDIO_DATA_CALLBACK_RESULT_INVALID;
    }
    int32_t writeSize = 0;
    if (!g_playFinishedFlag) {
        OneRenDerFrame(audioDataSize, &writeSize);
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
            "g_isRecord: %{public}s", g_isRecord ? "true" : "false");
        // 每次保存一次获取的buffer值
        if (audioDataSize != 0 && g_isRecord == true) {
            std::copy(g_playAudioData, g_playAudioData + writeSize,
                static_cast<char *>(g_playTotalAudioData) + g_playResultTotalSize);
            g_playResultTotalSize += writeSize;
        }
    }
    // 播放音频数据
    if (g_playAudioData != nullptr) {
        std::copy(g_playAudioData, g_playAudioData + audioDataSize, static_cast<char *>(audioData));
    }
    if (g_playFinishedFlag) {
        // 停止播放
        OH_AudioRenderer_Stop(audioRenderer);
        // 停止管线
        ResetAllIsResetTotalWriteAudioDataSize();
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
            "audioEditTest PlayAudioRendererOnWriteData g_playResultTotalSize is %{public}d", g_playResultTotalSize);
        CallBooleanCallback(g_playFinishedFlag);
        g_playFinishedFlag = false;
        if (g_totalBuff != nullptr) {
            free(g_totalBuff);
            g_totalBuff = nullptr;
        }
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REAL_TIME_PLAYING_TAG,
        "audioEditTest PlayAudioRendererOnWriteData g_playResultTotalSize: %{public}d, writeSize: %{public}d",
        g_playResultTotalSize, writeSize);
    return AUDIO_DATA_CALLBACK_RESULT_VALID;
}

void ReleaseExistingResources()
{
    if (audioRenderer) {
        // 释放播放器实例
        OH_AudioRenderer_Release(audioRenderer);
        audioRenderer = nullptr;
    }
    if (rendererBuilder) {
        // 释放构造器
        OH_AudioStreamBuilder_Destroy(rendererBuilder);
        rendererBuilder = nullptr;
    }
}