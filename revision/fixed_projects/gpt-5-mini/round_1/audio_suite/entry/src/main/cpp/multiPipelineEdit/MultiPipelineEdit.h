/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
#ifndef MULTIPIPELINEEDIT_H
#define MULTIPIPELINEEDIT_H
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "napi/native_api.h"
#include "ohaudio/native_audio_suite_base.h"
#include "ohaudio/native_audio_suite_engine.h"
#include "audioEffectNode/Input.h"
#include "NodeManager.h"
#include "PipelineManager.h"
#include "callback/RegisterCallback.h"
#include "audioSuiteError/AudioSuiteError.h"

#include <multimedia/player_framework/native_avdemuxer.h>
#include <multimedia/player_framework/native_avsource.h>
#include <multimedia/player_framework/native_avcodec_base.h>
#include <multimedia/player_framework/native_avformat.h>
#include <multimedia/player_framework/native_avbuffer.h>
#include <./utils/Utils.h>
// 对外napi接口
napi_value AudioEditNodeInitMultiPipeline(napi_env env, napi_callback_info info);
napi_value MultiAudioInAndOutInit(napi_env env, napi_callback_info info);
napi_value MultiPipelineEnvPrepare(napi_env env, napi_callback_info info);
napi_value MultiSetFormat(napi_env env, napi_callback_info info);
napi_value MultiSetEqualizerMode(napi_env env, napi_callback_info info);
napi_value MultiSetEqualizerFrequencyBandGains(napi_env env, napi_callback_info info);
napi_value MultiStartFieldEffect(napi_env env, napi_callback_info info);
napi_value MultiStartEnvEffect(napi_env env, napi_callback_info info);
napi_value MultiAddAudioSeparation(napi_env env, napi_callback_info info);
napi_value MultiAddNoiseReduction(napi_env env, napi_callback_info info);
napi_value MultiStartVBEffect(napi_env env, napi_callback_info info);
napi_value MultiSaveFileBuffer(napi_env env, napi_callback_info info);
napi_value MultiGetSecondOutputAudio(napi_env env, napi_callback_info info);
napi_value MultiDeleteSong(napi_env env, napi_callback_info info);
napi_value DestroyMultiPipeline(napi_env env, napi_callback_info info);

// 实现多管线功能函数
void MultiStoreTotalBuffToMap(const char *totalBuff, size_t size, const std::string &key);
OH_AudioSuite_Result MultiPipelineRenderFrame();
int32_t MultiWriteDataCallBack(OH_AudioNode *audioNode, void *userData, void *audioData, int32_t audioDataSize,
    bool *finished);
OH_AudioSuite_Result MultiSetParamsAndWriteData(OH_AudioNodeBuilder *builder,
    std::string inputId, OH_AudioNode_Type type);
void MultiCreateInputNode(napi_env env, const std::string &inputId, napi_value &napiValue,
    OH_AudioSuite_Result &result);
void MultiUpdateInputNode(OH_AudioSuite_Result &result, const UpdateInputNodeParams &params);
void MultiReadTrackSamples(OH_AVDemuxer *demuxer, uint32_t trackIndex, int bufferSize, std::atomic<bool> &isEnd,
    std::shared_ptr<PipelineManager> threadPipelineManager);
bool MultiGetAudioProperties(OH_AVFormat *trackFormat, int32_t &sampleRate, int32_t &channels, int32_t &bitsPerSample);
void MultiRunAudioThread(OH_AVDemuxer *demuxer, int32_t fileLength);
void MultiManageExistingOutputNodes(const std::string &inputId,
    const std::string &mixerId, OH_AudioSuite_Result &result, std::vector<Node> outPutNodes);
void MultiCreateAndConnectOutputNodes(const std::string &inputId, const std::string &outputId,
    OH_AudioSuite_Result &result);
void MultiManageOutputNodes(napi_env env, const std::string &inputId, const std::string &outputId,
    const std::string &mixerId, OH_AudioSuite_Result &result);
int MultiAddEffectNodeToNodeManager(std::string &inputNodeId, std::string &effectNodeId);
Node MultiCreateNodeByType(std::string uuid, OH_AudioNode_Type nodeType);

#endif