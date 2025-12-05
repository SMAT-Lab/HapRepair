/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "AissEffect.h"
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
const char *AISS_TAG = "[AudioEditTestApp_AISS_cpp]";

napi_value addAudioSeparation(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, AISS_TAG, "addAudioSeparation---IN");
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    std::string uuidStr;
    napi_status status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_0], uuidStr);
    std::string inputIdStr;
    status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_1], inputIdStr);
    std::string selectedNodeId;
    status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_2], selectedNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, AISS_TAG, "uuid:%{public}s, inputId:%{public}s,"
                 "selectedNodeId:%{public}s", uuidStr.c_str(), inputIdStr.c_str(), selectedNodeId.c_str());
    napi_value ret = nullptr;
    Node node = CreateNodeByType(uuidStr, OH_AudioNode_Type::EFFECT_MULTII_OUTPUT_NODE_TYPE_AUDIO_SEPARATION);
    if (node.physicalNode == nullptr) {
        napi_create_int64(env, AUDIOSUITE_ERROR_SYSTEM, &ret);
        return ret;
    }
    if (selectedNodeId.empty()) {
        int insertRes = AddEffectNodeToNodeManager(inputIdStr, uuidStr);
        if (insertRes == NODE_MANAGER_OPERATION_ERROR) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, AISS_TAG, "AddEffectNodeToNodeManager ERROR!");
            return ret;
        }
    } else {
        OH_AudioSuite_Result result =
            g_nodeManager->insertNode(uuidStr, selectedNodeId, Direction::LATER);
        if (result != AUDIOSUITE_SUCCESS) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, AISS_TAG,
                         "audioEditTest addAudioSeparation insertNode ERROR %{public}u", result);
        }
    }
    g_multiRenderFrameFlag = true;
    napi_create_int64(env, AUDIOSUITE_SUCCESS, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, AISS_TAG, "addAudioSeparation: operation success");
    return ret;
}

napi_value deleteAudioSeparation(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, AISS_TAG, "deleteAudioSeparation IN");
    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // 获取uuid
    std::string uuidStr;
    ParseNapiString(env, argv[NAPI_ARGV_INDEX_0], uuidStr);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, AISS_TAG, "uuid==%{public}s", uuidStr.c_str());

    OH_AudioSuite_Result result;
    napi_value napiValue = nullptr;
    result = g_nodeManager->removeNode(uuidStr);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, AISS_TAG, "audioEditTest removeNode ERROR:%{public}d", result);
    }
    napi_create_int64(env, result, &napiValue);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, AISS_TAG, "deleteAudioSeparation: operation success");
    return napiValue;
}