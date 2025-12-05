/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "EnvEffect.h"
#include "NoiseReduction.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include "napi/native_api.h"
#include "hilog/log.h"
#include "ohaudio/native_audio_suite_base.h"
#include "ohaudio/native_audio_suite_engine.h"
#include "NodeManager.h"
#include "callback/RegisterCallback.h"
#include "audioSuiteError/AudioSuiteError.h"
#include "audioEffectNode/Equailizer.h"
#include "audioEffectNode/EffectNode.h"
#include "audioEffectNode/Input.h"
#include "audioEffectNode/Output.h"
#include "realTimePlay/RealTimePlaying.h"
#include "multiPipelineEdit/MultiPipelineEdit.h"
#include "utils/Utils.h"
#include "./EffectNode.h"
#include "/utils/Constant.h"

const int GLOBAL_RESMGR = 0xFF00;
const char *ENV_TAG = "[AudioEditTestApp_ENV_cpp]";

napi_value startEnvEffect(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, ENV_TAG, "audioEditTest---startEnvEffect---IN");
    size_t argc = 4;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    napi_status status;
    std::string inputId;
    status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_0], inputId);
    std::string effectNodeId;
    status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_1], effectNodeId);
    unsigned int mode = 0;
    napi_get_value_uint32(env, argv[NAPI_ARGV_INDEX_2], &mode);
    std::string selectedNodeId;
    status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_3], selectedNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, ENV_TAG, "inputId:%{public}s, uuid:%{public}s, mode:%{public}d, "
                  "selectedNodeId:%{public}s", inputId.c_str(), effectNodeId.c_str(), mode, selectedNodeId.c_str());

    Node node;
    napi_value ret;
    OH_AudioSuite_Result result = createEnvNodeAndSetType(effectNodeId, mode, node);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, ENV_TAG, "createEnvNodeAndSetType ERROR!");
        napi_create_int64(env, result, &ret);
        return ret;
    }

    if (selectedNodeId.empty()) {
        int insertRes = AddEffectNodeToNodeManager(inputId, effectNodeId);
        if (insertRes == NODE_MANAGER_OPERATION_ERROR) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, ENV_TAG, "AddEffectNodeToNodeManager ERROR!");
            napi_create_int64(env, insertRes, &ret);
            return ret;
        }
    } else {
        result = g_nodeManager->insertNode(effectNodeId, selectedNodeId, Direction::LATER);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, ENV_TAG, "startEnvEffect insertNode ERROR!");
            napi_create_int64(env, result, &ret);
            return ret;
        }
    }

    napi_create_int64(env, AUDIOSUITE_SUCCESS, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, ENV_TAG, "startEnvEffect: operation success");
    return ret;
}

napi_value resetEnvEffect(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, ENV_TAG, "audioEditTest---resetEnvEffect---IN");
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    std::string inputId;
    ParseNapiString(env, argv[NAPI_ARGV_INDEX_0], inputId);
    std::string effectNodeId;
    ParseNapiString(env, argv[NAPI_ARGV_INDEX_1], effectNodeId);
    unsigned int mode = 0;
    napi_get_value_uint32(env, argv[NAPI_ARGV_INDEX_2], &mode);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, ENV_TAG, "inputId:%{public}s, uuid:%{public}s, mode:%{public}d",
                 inputId.c_str(), effectNodeId.c_str(), mode);

    OH_EnvironmentType type = GetEnvEnumByNumber(mode);
    napi_value ret;
    Node node = g_nodeManager->GetNodeById(effectNodeId);
    bool bypass = mode == 0;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_BypassEffectNode(node.physicalNode, bypass);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, ENV_TAG,
            "audioEditTest---resetEnvEffect OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        napi_create_int64(env, result, &ret);
        return ret;
    }
    if (bypass) {
        napi_create_int64(env, result, &ret);
        return ret;
    }
    result = OH_AudioSuiteEngine_SetEnvironmentType(node.physicalNode, type);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, ENV_TAG,
                     "OH_AudioSuiteEngine_SetEnvironmentType ERROR---%{public}d", result);
        napi_create_int64(env, result, &ret);
        return ret;
    }
    
    napi_create_int64(env, AUDIOSUITE_SUCCESS, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, ENV_TAG, "resetEnvEffect: operation success");
    return ret;
}

OH_AudioSuite_Result createEnvNodeAndSetType(std::string uuidStr, unsigned int mode, Node &node)
{
    OH_EnvironmentType type = GetEnvEnumByNumber(mode);
    node = CreateNodeByType(uuidStr, OH_AudioNode_Type::EFFECT_NODE_TYPE_ENVIRONMENT_EFFECT);
    if (node.physicalNode == nullptr) {
        return AUDIOSUITE_ERROR_SYSTEM;
    }
    OH_AudioSuite_Result result = AUDIOSUITE_SUCCESS;
    bool bypass = mode == 0;
    result = OH_AudioSuiteEngine_BypassEffectNode(node.physicalNode, bypass);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, ENV_TAG,
            "audioEditTest---startEnvEffect OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        return result;
    }
    if (bypass) {
        return result;
    }
    result = OH_AudioSuiteEngine_SetEnvironmentType(node.physicalNode, type);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, ENV_TAG,
                     "audioEditTest---OH_AudioSuiteEngine_SetEnvironmentType ERROR---%{public}d", result);
    }
    return result;
}