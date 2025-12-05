/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "hilog/log.h"
#include "Equailizer.h"
#include "./utils/Utils.h"
#include "./EffectNode.h"

const int GLOBAL_RESMGR = 0xFF00;
const char *EQUAILIZER_TAG = "[AudioEditTestApp_Equailizer_cpp]";

// 封装入参 OH_EqualizerMode
OH_EqualizerFrequencyBandGains SetEqualizerMode(int32_t equailizerMode)
{
    OH_EqualizerFrequencyBandGains eqMode;
    switch (equailizerMode) {
        case EQ_DEFAULT:
            eqMode = OH_EQUALIZER_PARAM_DEFAULT;
            break;
        case EQ_BALLADS:
            eqMode = OH_EQUALIZER_PARAM_BALLADS;
            break;
        case EQ_CHINESE_STYLE:
            eqMode = OH_EQUALIZER_PARAM_CHINESE_STYLE;
            break;
        case EQ_CLASSICAL:
            eqMode = OH_EQUALIZER_PARAM_CLASSICAL;
            break;
        case EQ_DANCE_MUSIC:
            eqMode = OH_EQUALIZER_PARAM_DANCE_MUSIC;
            break;
        case EQ_JAZZ:
            eqMode = OH_EQUALIZER_PARAM_JAZZ;
            break;
        case EQ_POP:
            eqMode = OH_EQUALIZER_PARAM_POP;
            break;
        case EQ_RB:
            eqMode = OH_EQUALIZER_PARAM_RB;
            break;
        case EQ_ROCK:
            eqMode = OH_EQUALIZER_PARAM_ROCK;
            break;
        default:
            eqMode = OH_EQUALIZER_PARAM_DEFAULT;
            break;
    }
    return eqMode;
}

napi_status GetEqModeParameters(
    napi_env env, napi_value *argv, unsigned int &equailizerMode, std::string &equailizerId, std::string &inputId)
{
    napi_status status = napi_get_value_uint32(env, argv[ARG_0], &equailizerMode);
    status = ParseNapiString(env, argv[ARG_1], equailizerId);
    status = ParseNapiString(env, argv[ARG_2], inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
        "audioEditTest GetEqModeParameters equailizerMode: %{public}d, equailizerId: %{public}s, inputId: %{public}s",
        equailizerMode, equailizerId.c_str(), inputId.c_str());
    return status;
}

napi_status GetEqBandGainsParameters(napi_env env, napi_value *argv, OH_EqualizerFrequencyBandGains &frequencyBandGains,
                                     EqBandGainsParams &params)
{
    // 遍历数组并打印每个元素
    for (uint32_t i = 0; i < EQUALIZER_BAND_NUM; ++i) {
        napi_value element;
        napi_get_element(env, argv[ARG_0], i, &element);
        unsigned int value;
        napi_get_value_uint32(env, element, &value);
        frequencyBandGains.gains[i] = value;
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
                     "audioEditTest getEqBandGainsParamters"
                     " element at index %{public}d is %{public}d",
                     i, frequencyBandGains.gains[i]);
    }
    napi_status status = ParseNapiString(env, argv[ARG_1], params.equailizerId);
    status = ParseNapiString(env, argv[ARG_2], params.inputId);
    ParseNapiString(env, argv[ARG_3], params.selectedNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
                 "audioEditTest equailizerId: %{public}s, inputId: %{public}s, selectedNodeId: %{public}s",
                 params.equailizerId.c_str(), params.inputId.c_str(), params.selectedNodeId.c_str());
    return status;
}

Node GetOrCreateEqualizerNodeByMode(std::string& equailizerId, std::string& inputId)
{
    Node eqNode = g_nodeManager->GetNodeById(equailizerId);
    if (!eqNode.physicalNode) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
            "audioEditTest GetOrCreateEqualizerNodeByMode create");
        eqNode.id = equailizerId;
        eqNode.type = OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER;
        g_nodeManager->createNode(equailizerId, OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER);
        eqNode = g_nodeManager->GetNodeById(equailizerId);
        int32_t result = AddEffectNodeToNodeManager(inputId, equailizerId);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
            "audioEditTest addEffectNodeManager result: %{public}d", result);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            eqNode.physicalNode = nullptr; // 标记为失败
        }
    }
    return eqNode;
}

Node GetOrCreateEqualizerNodeByGains(std::string& equailizerId, std::string& inputId, std::string& selectedNodeId)
{
    Node eqNode = g_nodeManager->GetNodeById(equailizerId);
    if (!eqNode.physicalNode) {
        // 创建均衡器节点
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
            "audioEditTest GetOrCreateEqualizerNodeByGains create");
        eqNode.id = equailizerId;
        eqNode.type = OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER;
        g_nodeManager->createNode(equailizerId, OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER);
        // 获取效果节点
        eqNode = g_nodeManager->GetNodeById(equailizerId);
        if (selectedNodeId.empty()) {
            int result = AddEffectNodeToNodeManager(inputId, equailizerId);
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
                "audioEditTest AddEffectNodeToNodeManager AddEffectNodeToNodeManager result: %{public}d",
                result);
            if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
                eqNode.physicalNode = nullptr; // 标记为失败
            }
        } else {
            int result = g_nodeManager->insertNode(equailizerId, selectedNodeId, Direction::LATER);
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, EQUAILIZER_TAG,
                "audioEditTest AddEffectNodeToNodeManager insertNode result: %{public}d", result);
            if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
                eqNode.physicalNode = nullptr; // 标记为失败
            }
        }
    }
    return eqNode;
}