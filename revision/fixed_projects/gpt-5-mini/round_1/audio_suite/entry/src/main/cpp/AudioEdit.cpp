/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <string>
#include <map>
#include <algorithm>
#include <sys/stat.h>
#include "napi/native_api.h"
#include <unistd.h>
#include "hilog/log.h"
#include "ohaudio/native_audiocapturer.h"
#include "ohaudio/native_audiorenderer.h"
#include "ohaudio/native_audiostreambuilder.h"
#include "ohaudio/native_audiostream_base.h"
#include "ohaudio/native_audio_suite_base.h"
#include "ohaudio/native_audio_suite_engine.h"
#include "NodeManager.h"
#include "audioEffectNode/EffectNode.h"
#include "audioEffectNode/VoiceBeautifier.h"
#include <iomanip>
#include <fstream>
#include <filemanagement/file_uri/oh_file_uri.h>
#include "callback/RegisterCallback.h"
#include "audioSuiteError/AudioSuiteError.h"
#include "audioEffectNode/Input.h"
#include "audioEffectNode/Output.h"
#include "audioEffectNode/EffectNode.h"
#include "audioEffectNode/ParseNapiParam.h"
#include "audioEffectNode/Equailizer.h"
#include "audioEffectNode/SoundField.h"
#include "callback/RegisterCallback.h"
#include "audioEffectNode/NoiseReduction.h"
#include "audioEffectNode/EnvEffect.h"
#include "audioEffectNode/AissEffect.h"
#include "./utils/Utils.h"
#include "realTimePlay/RealTimePlaying.h"
#include "audioSuiteError/AudioSuiteError.h"
#include "multiPipelineEdit/MultiPipelineEdit.h"

#include <multimedia/player_framework/native_avdemuxer.h>
#include <multimedia/player_framework/native_avsource.h>
#include <multimedia/player_framework/native_avcodec_base.h>
#include <multimedia/player_framework/native_avformat.h>
#include <multimedia/player_framework/native_avbuffer.h>
#include <fcntl.h>

const int GLOBAL_RESMGR = 0xFF00;
const char *TAG = "[AudioEditTestApp_AudioEdit_cpp]";
void *g_aissTapAudioData = (char *)malloc(1024 * 1024 * 100);

const int MAX_PLAY_RESULT_BUFFER_SIZE = 1024 * 1024 * 1024;

const int SAMPLINGRATE_MULTI = 20;
const int CHANNELCOUNT_MULTI = 1000;
const int BITSPERSAMPLE_MULTI = 8;
const int INPUTNODES_SIZE2 = 2;

static napi_ref callbackStringArrayRef = nullptr;
static napi_value RegisterAudioFormatCallback(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1];
    napi_value callback;
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    // 创建全局引用
    napi_create_reference(env, args[0], 1, &callbackStringArrayRef);

    // 创建线程安全函数
    napi_value global;
    napi_get_global(env, &global);
    napi_get_reference_value(env, callbackStringArrayRef, &callback);
    napi_value name;
    napi_status status = napi_create_string_utf8(env, "CallStringArrayCallback", NAPI_AUTO_LENGTH, &name);
    napi_create_threadsafe_function(env, callback, NULL, name, 1, 1, NULL, NULL, NULL, CallStringArrayThread,
        &tsfnStringArray);

    napi_value result;
    napi_get_undefined(env, &result);
    return result;
}

static napi_value AudioEditNodeInit(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest AudioEditNodeInit start");
    size_t argc = 1;
    napi_value argv[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // 解析工作模式

    // 创建引擎
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_Create(&g_audioSuiteEngine);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest OH_AudioEditEngine_Create result: %{public}d",
        static_cast<int>(result));
    // 根据入参判断当前的workmode
    unsigned int mode = -1;
    napi_get_value_uint32(env, argv[0], &mode);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---AudioEditNodeInit--workMode==%{public}d",
        mode);
    OH_AudioSuite_PipelineWorkMode workMode;
    if (mode == OH_AudioSuite_PipelineWorkMode::AUDIOSUITE_PIPELINE_EDIT_MODE) {
        workMode = OH_AudioSuite_PipelineWorkMode::AUDIOSUITE_PIPELINE_EDIT_MODE;
    } else if (mode == OH_AudioSuite_PipelineWorkMode::AUDIOSUITE_PIPELINE_REALTIME_MODE) {
        workMode = OH_AudioSuite_PipelineWorkMode::AUDIOSUITE_PIPELINE_REALTIME_MODE;
    } else {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,
            "audioEditTest OH_AudioEditEngine_CreatePipeline workMode error: %{public}d", workMode);
        return nullptr;
    }
    // 创建管线
    result = OH_AudioSuiteEngine_CreatePipeline(g_audioSuiteEngine, &g_audioSuitePipeline, workMode);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG,
        "audioEditTest OH_AudioEditEngine_CreatePipeline result: %{public}d", static_cast<int>(result));
    // 实例化NodeManager
    g_nodeManager = std::make_shared<NodeManager>(g_audioSuitePipeline);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest createNodeManager result: %{public}d",
        static_cast<int>(g_nodeManager->getAllNodes().size()));

    napi_value napiValue;
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

// 释放内存
static void Clear()
{
    // 释放map内存
    g_writeDataBufferMap.clear();
    for (auto &pair : g_userDataMap) {
        delete pair.second; // 删除指针指向的对象
    }
    g_userDataMap.clear();
}

static napi_value AudioEditDestory(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest AudioEditDestory start");
    Clear();
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_DestroyPipeline(g_audioSuitePipeline);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG,
        "audioEditTest OH_audioSuiteEngine_DestroyPipeline result: %{public}d", static_cast<int>(result));
    result = OH_AudioSuiteEngine_Destroy(g_audioSuiteEngine);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest OH_audioSuiteEngine_Destroy result: %{public}d",
        static_cast<int>(result));
    napi_value napiValue;
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

static napi_value SetFormat(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest SetFormat start");
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // 获取通道数
    unsigned int channels;
    napi_get_value_uint32(env, argv[ARG_0], &channels);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest SetFormat channels is %{public}d", channels);
    // 获取采样率
    unsigned int sampleRate;
    napi_get_value_uint32(env, argv[ARG_1], &sampleRate);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest SetFormat sampleRate is %{public}d", sampleRate);
    // 获取位深
    unsigned int bitsPerSample;
    napi_get_value_uint32(env, argv[ARG_2], &bitsPerSample);
    ConvertBitsPerSample(bitsPerSample);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest SetFormat bitsPerSample is %{public}d",
        bitsPerSample);

    // 设置采样率
    g_audioFormatOutput.samplingRate = SetSamplingRate(sampleRate);
    // 设置声道
    g_audioFormatOutput.channelCount = channels;
    g_audioFormatOutput.channelLayout = SetChannelLayout(channels);
    // 设置位深
    g_audioFormatOutput.sampleFormat = SetSampleFormat(bitsPerSample);
    // 设置编码格式
    g_audioFormatOutput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;
    const std::vector<Node> outPutNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_SetAudioFormat(outPutNodes[0].physicalNode, &g_audioFormatOutput);
    napi_value napiValue;
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

// 导入音频调用
static napi_value AudioInAndOutInit(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest AudioInAndOutInit start");
    size_t argc = 5;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    AudioParams params;
    napi_status status = ParseArguments(env, argv, params);
    if (status != napi_ok) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(status));
    }
    OH_AVSource *source = OH_AVSource_CreateWithFD(params.fd, 0, params.fileLength);
    if (source == nullptr) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    OH_AVFormat *trackFormat = OH_AVSource_GetTrackFormat(source, 0);
    if (trackFormat == nullptr) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    // 采样率，声道，位深
    int32_t sampleRate;
    int32_t channels;
    int32_t bitsPerSample;
    if (!GetAudioProperties(trackFormat, sampleRate, channels, bitsPerSample)) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    std::vector<std::string> audioFormat = {
        std::to_string(sampleRate), std::to_string(channels), std::to_string(bitsPerSample)
    };
    CallStringArrayCallback(audioFormat);
    // 为资源实例创建对应的解封器
    OH_AVDemuxer *demuxer = OH_AVDemuxer_CreateWithSource(source);
    if (demuxer == nullptr) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    RunAudioThread(demuxer, params.fileLength);
    napi_value napiValue;
    OH_AudioSuite_Result result;
    Node inputNode = g_nodeManager->GetNodeById(params.inputId);
    if (inputNode.id.empty()) {
        CreateInputNode(env, params.inputId, napiValue, result);
    } else {
        UpdateInputNodeParams updateInputNodeParams;
        updateInputNodeParams.inputId = params.inputId;
        updateInputNodeParams.channels = channels;
        updateInputNodeParams.sampleRate = sampleRate;
        updateInputNodeParams.bitsPerSample = bitsPerSample;
        UpdateInputNode(napiValue, result, updateInputNodeParams);
        return ReturnResult(env, static_cast<AudioSuiteResult>(result));
    }
    ManageOutputNodes(env, params.inputId, params.outputId, params.mixerId, result);
    return ReturnResult(env, static_cast<AudioSuiteResult>(result));
}

OH_AudioSuite_Result DeleteNodeOfSong(Node &node, int size)
{
    OH_AudioSuite_Result result;
    Node nextNode;
    if (size > INPUTNODES_SIZE2) {
        while (node.type != OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER) {
            nextNode = g_nodeManager->GetNodeById(node.nextNodeId);
            result = g_nodeManager->removeNode(node.id);
            return result;
            node = nextNode;
        }
    } else if (size == INPUTNODES_SIZE2) {
        while (node.type != OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT) {
            nextNode = g_nodeManager->GetNodeById(node.nextNodeId);
            result = g_nodeManager->removeNode(node.id);
            return result;
            node = nextNode;
        }
    } else {
        while (!node.id.empty()) {
            nextNode = g_nodeManager->GetNodeById(node.nextNodeId);
            result = g_nodeManager->removeNode(node.id);
            return result;
            node = nextNode;
        }
    }
}

// 删除音频
static napi_value DeleteSong(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest DeleteSong start");

    OH_AudioSuite_Result result;
    napi_value napiValue;
    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    // 获取inputId参数
    std::string inputId;
    napi_status status = ParseNapiString(env, argv[0], inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest DeleteSong inputId is %{public}s",
        inputId.c_str());
    
    const std::vector<Node> inputNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest DeleteSong inputNodes length is %{public}d",
        static_cast<int>(inputNodes.size()));

    Node node = g_nodeManager->GetNodeById(inputId);
    Node nextNode;
    if (node.id.empty()) {
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        return napiValue;
    }
    
    if (inputNodes.size() <= 0) {
        napi_create_int64(env, static_cast<int>(-1), &napiValue);
        return napiValue;
    } else {
        result = DeleteNodeOfSong(node, inputNodes.size());
    }

    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

// 删除节点
static napi_value DeleteNode(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest DeleteNode start");

    OH_AudioSuite_Result result;
    napi_value napiValue;
    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    // 获取nodeId参数
    std::string nodeId;
    napi_status status = ParseNapiString(env, argv[0], nodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest DeleteNode nodeId is %{public}s",
        nodeId.c_str());
    
    result = g_nodeManager->removeNode(nodeId);

    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

// 设置均衡器模式方法
static napi_value SetEquailizerMode(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest SetEquailizerMode start");
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    unsigned int equailizerMode = -1;
    std::string equailizerId;
    std::string inputId;
    napi_status status = GetEqModeParameters(env, argv, equailizerMode, equailizerId, inputId);
    if (status != napi_ok) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(AudioSuiteResult::DEMO_PARAMETER_ANALYSIS_ERROR));
    }

    // 创建均衡器效果节点
    Node eqNode = GetOrCreateEqualizerNodeByMode(equailizerId, inputId);
    if (!eqNode.physicalNode) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(AudioSuiteResult::DEMO_CREATE_NODE_ERROR));
    }
    bool bypass = equailizerMode == 0;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_BypassEffectNode(eqNode.physicalNode, bypass);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,
            "audioEditTest---SetEquailizerMode OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        return ReturnResult(env, static_cast<AudioSuiteResult>(result));
    }
    if (bypass) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(result));
    }
    result =
        OH_AudioSuiteEngine_SetEqualizerFrequencyBandGains(eqNode.physicalNode, SetEqualizerMode(equailizerMode));
    return ReturnResult(env, static_cast<AudioSuiteResult>(result));
}

// 设置均衡器频带增益
static napi_value SetEqualizerFrequencyBandGains(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest SetEqualizerFrequencyBandGains start");
    size_t argc = 4;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    OH_EqualizerFrequencyBandGains frequencyBandGains;
    EqBandGainsParams params;
    napi_status status = GetEqBandGainsParameters(env, argv, frequencyBandGains, params);
    if (status != napi_ok) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(AudioSuiteResult::DEMO_PARAMETER_ANALYSIS_ERROR));
    }
    // 创建均衡器效果节点
    Node eqNode = GetOrCreateEqualizerNodeByGains(params.equailizerId, params.inputId, params.selectedNodeId);
    if (!eqNode.physicalNode) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(AudioSuiteResult::DEMO_CREATE_NODE_ERROR));
    }

    OH_AudioSuite_Result result =
        OH_AudioSuiteEngine_SetEqualizerFrequencyBandGains(eqNode.physicalNode, frequencyBandGains);
    return ReturnResult(env, static_cast<AudioSuiteResult>(result));
}

static napi_value SaveFileBuffer(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest SaveFileBuffer start");
    ResetAllIsResetTotalWriteAudioDataSize();
    RenDerFrame();
    Clear();

    napi_value napiValue = nullptr;
    void *arrayBufferData = nullptr;
    napi_status status = napi_create_arraybuffer(env, g_totalSize, &arrayBufferData, &napiValue);
    if (status != napi_ok || arrayBufferData == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG,
            "audioEditTest OH_AudioSuiteEngine_RenderFrame status: %{public}d", static_cast<int>(status));
        if (g_totalBuff != nullptr) {
            free(g_totalBuff);
            g_totalBuff = nullptr;
        }
        // 创建 ArrayBuffer 失败， 返回一个大小为 0 的ArrayBuffer
        napi_create_arraybuffer(env, 0, &arrayBufferData, &napiValue);
        return napiValue;
    } else {
        std::copy(g_totalBuff, g_totalBuff + g_totalSize, static_cast<char *>(arrayBufferData));
        if (g_totalBuff != nullptr) {
            free(g_totalBuff);
            g_totalBuff = nullptr;
        }
        return napiValue;
    }
}

static napi_value startVBEffect(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---startVBEffect---IN");
    size_t argc = 4;
    napi_value argv[4] = {nullptr, nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // inputId
    std::string inputId;
    napi_status status = ParseNapiString(env, argv[ARG_0], inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---startVBEffect---inputId==%{public}s",
                 inputId.c_str());
    // 获取二参、美化类型
    int mode = -1;
    napi_get_value_int32(env, argv[ARG_1], &mode);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---startVBEffect--mode==%{public}zd", mode);
    // 获取三参、效果节点id
    std::string voiceBeautifierId;
    status = ParseNapiString(env, argv[ARG_2], voiceBeautifierId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---uuid==%{public}s", voiceBeautifierId.c_str());
    // 获取当前选中的节点id
    std::string selectNodeId;
    status = ParseNapiString(env, argv[ARG_3], selectNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---startVBEffect---selectNodeId==%{public}s",
                 selectNodeId.c_str());
     //调用添加美化效果节点接口
    napi_value ret;
    int result = AddVBEffectNode(inputId, mode, voiceBeautifierId, selectNodeId);

    napi_create_int64(env, result, &ret);
    return ret;
}
static napi_value resetVBEffect(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---resetVBEffect---IN");
    size_t argc = 3;
    napi_value argv[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    int mode = -1;
    std::string inputId;
    std::string voiceBeautifierId;
    //解析参数
    napi_status status = getResetVBParameters(env, argv, inputId, mode, voiceBeautifierId);
    if (status != napi_ok) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(AudioSuiteResult::DEMO_PARAMETER_ANALYSIS_ERROR));
    }
    napi_value ret;
    int result = ModifyVBEffectNode(inputId, mode, voiceBeautifierId);
    napi_create_int64(env, result, &ret);
    return ret;
}

static napi_status ParseFieldEffectParams(napi_env env, napi_callback_info info, FieldEffectParams& params)
{
    size_t argc = 4;
    napi_value argv[4] = {nullptr, nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    napi_status status = ParseNapiString(env, argv[ARG_0], params.inputId);
    napi_get_value_uint32(env, argv[ARG_1], &params.mode);
    status = ParseNapiString(env, argv[ARG_2], params.fieldEffectId);
    status = ParseNapiString(env, argv[ARG_3], params.selectedNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG,
        "audioEditTest FieldEffect inputId==%{public}s mode==%{public}zd, " \
        "fieldEffectId==%{public}s, selectedNodeId==%{public}s", \
        params.inputId.c_str(),
        params.mode,
        params.fieldEffectId.c_str(),
        params.selectedNodeId.c_str());
    return status;
}

static napi_value startFieldEffect(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---startFieldEffect start");
    FieldEffectParams params;
    napi_status status = ParseFieldEffectParams(env, info, params);
    OH_SoundFieldType type = getSoundFieldTypeByNum(params.mode);
    napi_value ret;
    Node node = CreateNodeByType(params.fieldEffectId, OH_AudioNode_Type::EFFECT_NODE_TYPE_SOUND_FIELD);
    bool bypass = params.mode == 0;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_BypassEffectNode(node.physicalNode, bypass);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,
            "audioEditTest---startFieldEffect OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        napi_create_int64(env, result, &ret);
        return ret;
    }
    if (bypass) {
        napi_create_int64(env, result, &ret);
        return ret;
    }
    result = OH_AudioSuiteEngine_SetSoundFieldType(node.physicalNode, type);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,
            "audioEditTest startFieldEffect OH_AudioEditEngine_SetSoundFiledType ERROR!");
        napi_create_int64(env, result, &ret);
        return ret;
    }
    if (params.selectedNodeId.empty()) {
        int res = AddEffectNodeToNodeManager(params.inputId, params.fieldEffectId);
        if (res != 0) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,
                "audioEditTest startFieldEffect AddEffectNodeToNodeManager ERROR!");
            napi_create_int64(env, res, &ret);
            return ret;
        }
    } else {
        result = g_nodeManager->insertNode(params.fieldEffectId, params.selectedNodeId, Direction::LATER);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "audioEditTest startFieldEffect insertNode ERROR!");
            napi_create_int64(env, result, &ret);
            return ret;
        }
    }

    napi_create_int64(env, result, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest startFieldEffect: operation success");
    return ret;
}

static napi_value resetFieldEffect(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest resetFieldEffect start");
    size_t argc = 3;
    napi_value argv[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    std::string inputId;
    napi_status status = ParseNapiString(env, argv[ARG_1], inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest resetFieldEffect inputId is %{public}s",
        inputId.c_str());

    // 获取二参
    unsigned int mode = -1;
    napi_get_value_uint32(env, argv[ARG_2], &mode);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest resetFieldEffect mode is %{public}zd", mode);

    // 获取三参
    std::string fieldEffectId;
    status = ParseNapiString(env, argv[ARG_3], fieldEffectId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest fieldEffectId is %{public}s",
        fieldEffectId.c_str());

    OH_SoundFieldType type = getSoundFieldTypeByNum(mode);

    napi_value ret;
    Node node = g_nodeManager->GetNodeById(fieldEffectId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest get node is %{public}s", node.id.c_str());
    bool bypass = mode == 0;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_BypassEffectNode(node.physicalNode, bypass);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,
            "audioEditTest---resetFieldEffect OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        napi_create_int64(env, result, &ret);
        return ret;
    }
    if (bypass) {
        napi_create_int64(env, result, &ret);
        return ret;
    }
    result = OH_AudioSuiteEngine_SetSoundFieldType(node.physicalNode, type);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG,
            "audioEditTest OH_AudioSuiteEngine_SetSoundFieldType ERROR %{public}zd", result);
        napi_create_int64(env, result, &ret);
        return ret;
    }

    napi_create_int64(env, result, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest resetFieldEffect: operation success");
    return ret;
}

static napi_value getAudioOfTap(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---getAudioOfTap---IN");

    napi_value napiValue = nullptr;
    void *data;
    napi_create_arraybuffer(env, g_tapDataTotalSize, &data, &napiValue);
    std::copy(
        reinterpret_cast<const char*>(g_tapTotalBuff),
        reinterpret_cast<const char*>(g_tapTotalBuff) + g_tapDataTotalSize,
        reinterpret_cast<char*>(data));
    std::fill(
        reinterpret_cast<char*>(g_tapTotalBuff),
        reinterpret_cast<char*>(g_tapTotalBuff) + g_tapDataTotalSize,
        0);
    g_tapDataTotalSize = 0;
    return napiValue;
}

// 音频播放 -------------------------------------
static napi_ref callbackAudioRendererRef = nullptr;

// 注册回调，获取音频播放的finished的值
static napi_value RegisterFinishedCallback(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1];
    napi_value callback;
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    // 创建全局引用
    napi_create_reference(env, args[0], 1, &callbackAudioRendererRef);

    // 创建线程安全函数
    napi_value global;
    napi_get_global(env, &global);
    napi_get_reference_value(env, callbackAudioRendererRef, &callback);
    napi_value name;
    napi_status status = napi_create_string_utf8(env, "CallBooleanCallback", NAPI_AUTO_LENGTH, &name);
    napi_create_threadsafe_function(env, callback, NULL, name, 1, 1, NULL, NULL, NULL, CallBoolThread, &tsfnBoolean);

    napi_value result;
    napi_get_undefined(env, &result);
    return result;
}

static napi_value Record(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest Record start");
    g_isRecord = true;
    return nullptr;
}

static napi_value RealTimeSaveFileBuffer(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest RealTimeSaveFileBuffer start");
    ResetAllIsResetTotalWriteAudioDataSize();
    g_isRecord = false;
    napi_value napiValue = nullptr;
    void *arrayBufferData = nullptr;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG,
        "audioEditTest RealTimeSaveFileBuffer g_playResultTotalSize is %{public}d", g_playResultTotalSize);
    napi_status status = napi_create_arraybuffer(env, g_playResultTotalSize, &arrayBufferData, &napiValue);
    if (status != napi_ok || arrayBufferData == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest napi_create_arraybuffer status: %{public}d",
            static_cast<int>(status));
        g_playResultTotalSize = 0;
        if (g_playTotalAudioData != nullptr) {
            free(g_playTotalAudioData);
            g_playTotalAudioData = nullptr;
        }
        // 创建 ArrayBuffer 失败，返回一个大小为 0 的ArrayBuffer
        napi_create_arraybuffer(env, 0, &arrayBufferData, &napiValue);
        return napiValue;
    } else {
        std::copy(g_playTotalAudioData, g_playTotalAudioData + g_playResultTotalSize,
            static_cast<char *>(arrayBufferData));
        if (g_playTotalAudioData != nullptr) {
            free(g_playTotalAudioData);
            g_playTotalAudioData = nullptr;
        }
        g_playResultTotalSize = 0;
        return napiValue;
    }
}

static napi_value AudioRendererInit(napi_env env, napi_callback_info info)
{
    ReleaseExistingResources();
    // 创建构造器
    OH_AudioStream_Type type = OH_AudioStream_Type::AUDIOSTREAM_TYPE_RENDERER;
    OH_AudioStreamBuilder_Create(&rendererBuilder, type);

    // 获取位深
    int32_t bitsPerSample = 0;
    OH_AudioStream_SampleFormat streamSampleFormat;
    GetBitsPerSampleAndStreamFormat(g_audioFormatOutput, &bitsPerSample, &streamSampleFormat);

    // 设置音频采样率。
    OH_AudioStreamBuilder_SetSamplingRate(rendererBuilder, g_audioFormatOutput.samplingRate);
    // 设置音频声道。
    OH_AudioStreamBuilder_SetChannelCount(rendererBuilder, g_audioFormatOutput.channelCount);
    // 设置音频采样格式。
    OH_AudioStreamBuilder_SetSampleFormat(rendererBuilder, streamSampleFormat);
    // 设置音频流的编码类型。
    OH_AudioStreamBuilder_SetEncodingType(rendererBuilder, AUDIOSTREAM_ENCODING_TYPE_RAW);
    // 设置输出音频流的工作场景。
    OH_AudioStreamBuilder_SetRendererInfo(rendererBuilder, AUDIOSTREAM_USAGE_MUSIC);
    // 设置 audioDataSize 长度 （待播放的数据大小）
    g_playDataSize = SAMPLINGRATE_MULTI * g_audioFormatOutput.samplingRate *
        g_audioFormatOutput.channelCount / CHANNELCOUNT_MULTI * bitsPerSample / BITSPERSAMPLE_MULTI;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG,
        "audioEditTest AudioRendererInit g_playDataSize: %{public}d, samplingRate: %{public}d, "
        "channelCount: %{public}d, bitsPerSample: %{public}d",
        g_playDataSize, g_audioFormatOutput.samplingRate, g_audioFormatOutput.channelCount, bitsPerSample);
    OH_AudioStreamBuilder_SetFrameSizeInCallback(rendererBuilder, g_playDataSize);

    // 配置写入音频数据回调函数。
    OH_AudioRenderer_OnWriteDataCallback rendererCallbacks = PlayAudioRendererOnWriteData;
    OH_AudioStreamBuilder_SetRendererWriteDataCallback(rendererBuilder, rendererCallbacks, nullptr);

    // create OH_AudioRenderer
    OH_AudioStreamBuilder_GenerateRenderer(rendererBuilder, &audioRenderer);
    return nullptr;
}

static napi_value AudioRendererDestory(napi_env env, napi_callback_info info)
{
    napi_value napiValue = nullptr;
    if (audioRenderer) {
        // 释放播放实例
        OH_AudioStream_Result result = OH_AudioRenderer_Release(audioRenderer);
        // 释放构造器
        result = OH_AudioStreamBuilder_Destroy(rendererBuilder);
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        audioRenderer = nullptr;
        rendererBuilder = nullptr;
    }
    return napiValue;
}

// 开始播放
static napi_value AudioRendererStart(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "g_writeDataBufferMap size %{public}d",
        g_writeDataBufferMap.size());
    ProcessPipeline();
    g_playTotalAudioData = (char *)malloc(MAX_PLAY_RESULT_BUFFER_SIZE);
    // start
    OH_AudioRenderer_Start(audioRenderer);
    return nullptr;
}

// 暂停播放
static napi_value AudioRendererPause(napi_env env, napi_callback_info info)
{
    // pause
    OH_AudioRenderer_Pause(audioRenderer);
    return nullptr;
}

// 停止播放
static napi_value AudioRendererStop(napi_env env, napi_callback_info info)
{
    // stop
    OH_AudioRenderer_Stop(audioRenderer);
    // 停止管线
    OH_AudioSuiteEngine_StopPipeline(g_audioSuitePipeline);
    return nullptr;
}

// 获取播放状态
static napi_value GetRendererState(napi_env env, napi_callback_info info)
{
    OH_AudioStream_State state;
    OH_AudioRenderer_GetCurrentState(audioRenderer, &state);
    napi_value sum;
    napi_create_int32(env, state, &sum);

    return sum;
}

// 是否重置totalWriteAudioDataSize
static napi_value ResetTotalWriteAudioDataSize(napi_env env, napi_callback_info info)
{
    // 写入音频的buffer，重头开始
    ResetAllIsResetTotalWriteAudioDataSize();
    // 报错音频的也重头开始保存
    g_playResultTotalSize = 0;
    return nullptr;
}

//获取效果节点options
static napi_value getOptions(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    napi_value napiValue;
    
    //获取nodeId
    std::string nodeId;
    ParseNapiString(env, argv[0], nodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "getOptions nodeId is %{public}s", nodeId.c_str());
    Node node = g_nodeManager->GetNodeById(nodeId);
    //根据不同效果类型获取效果参数
    std::string type = g_nodeManager->GetOptionsByType(node);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "getOptions type is %{public}s", type.c_str());
    napi_create_string_utf8(env, type.c_str(), NAPI_AUTO_LENGTH, &napiValue);
    return napiValue;
}

static napi_value getEffectNodeList(napi_env env, napi_callback_info info)
{
    // 返回 JS 数组
    return GetSupportedAudioNodeTypes(env);
}

const std::vector<napi_property_descriptor> multiPipelineDescriptors = {
    {"audioEditNodeInitMultiPipeline", nullptr, AudioEditNodeInitMultiPipeline,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiAudioInAndOutInit", nullptr, MultiAudioInAndOutInit,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiPipelineEnvPrepare", nullptr, MultiPipelineEnvPrepare,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiSetFormat", nullptr, MultiSetFormat,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiSetEqualizerMode", nullptr, MultiSetEqualizerMode,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiSetEqualizerFrequencyBandGains", nullptr, MultiSetEqualizerFrequencyBandGains,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiStartFieldEffect", nullptr, MultiStartFieldEffect,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiStartEnvEffect", nullptr, MultiStartEnvEffect,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiAddAudioSeparation", nullptr, MultiAddAudioSeparation,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiAddNoiseReduction", nullptr, MultiAddNoiseReduction,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiStartVBEffect", nullptr, MultiStartVBEffect,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiSaveFileBuffer", nullptr, MultiSaveFileBuffer,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiGetSecondOutputAudio", nullptr, MultiGetSecondOutputAudio,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"multiDeleteSong", nullptr, MultiDeleteSong,
        nullptr, nullptr, nullptr, napi_default, nullptr},
    {"destroyMultiPipeline", nullptr, DestroyMultiPipeline,
        nullptr, nullptr, nullptr, napi_default, nullptr}
};

EXTERN_C_START static napi_value Init(napi_env env, napi_value exports)
{
    std::vector<napi_property_descriptor> desc = {
        {"record", nullptr, Record, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioRendererInit", nullptr, AudioRendererInit, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioRendererDestory", nullptr, AudioRendererDestory, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioRendererStart", nullptr, AudioRendererStart, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioRendererPause", nullptr, AudioRendererPause, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioRendererStop", nullptr, AudioRendererStop, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getRendererState", nullptr, GetRendererState, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"registerFinishedCallback", nullptr, RegisterFinishedCallback, nullptr, nullptr, nullptr, napi_default,
            nullptr},
        {"resetTotalWriteAudioDataSize", nullptr, ResetTotalWriteAudioDataSize, nullptr, nullptr, nullptr, napi_default,
            nullptr},
        {"realTimeSaveFileBuffer", nullptr, RealTimeSaveFileBuffer, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioEditNodeInit", nullptr, AudioEditNodeInit, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioInAndOutInit", nullptr, AudioInAndOutInit, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"audioEditDestory", nullptr, AudioEditDestory, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setFormat", nullptr, SetFormat, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setEquailizerMode", nullptr, SetEquailizerMode, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setEqualizerFrequencyBandGains", nullptr, SetEqualizerFrequencyBandGains, nullptr, nullptr, nullptr,
            napi_default, nullptr},
        {"saveFileBuffer", nullptr, SaveFileBuffer, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"startFieldEffect", nullptr, startFieldEffect, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"startVBEffect", nullptr, startVBEffect, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"addAudioSeparation", nullptr, addAudioSeparation, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"addNoiseReduction", nullptr, addNoiseReduction, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetFieldEffect", nullptr, resetFieldEffect, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetVBEffect", nullptr, resetVBEffect, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"deleteNoiseReduction", nullptr, deleteNoiseReduction, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"deleteSong", nullptr, DeleteSong, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"deleteAudioSeparation", nullptr, deleteAudioSeparation, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getAudioOfTap", nullptr, getAudioOfTap, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"startEnvEffect", nullptr, startEnvEffect, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetEnvEffect", nullptr, resetEnvEffect, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"deleteNode", nullptr, DeleteNode, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"registerAudioFormatCallback", nullptr, RegisterAudioFormatCallback, nullptr, nullptr, nullptr,
            napi_default, nullptr},
        {"getOptions", nullptr, getOptions, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getEffectNodeList", nullptr, getEffectNodeList, nullptr, nullptr, nullptr, napi_default, nullptr}};
    desc.insert(desc.end(), multiPipelineDescriptors.begin(), multiPipelineDescriptors.end());
    napi_define_properties(env, exports, desc.size(), desc.data());
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void *)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void) { napi_module_register(&demoModule); }
