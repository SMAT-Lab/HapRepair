/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

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
const char *NR_TAG = "[AudioEditTestApp_NR_cpp]";

napi_value addNoiseReduction(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, NR_TAG, "audioEditTest---addNoiseReduction IN");
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    std::string uuidStr;
    napi_status status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_0], uuidStr);
    std::string inputIdStr;
    status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_1], inputIdStr);
    std::string selectNodeId;
    status = ParseNapiString(env, argv[NAPI_ARGV_INDEX_2], selectNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, NR_TAG, "uuid:%{public}s, inputId:%{public}s, "
                 "selectNodeId:%{public}s", uuidStr.c_str(), inputIdStr.c_str(), selectNodeId.c_str());

    napi_value ret = nullptr;
    Node node = CreateNodeByType(uuidStr, OH_AudioNode_Type::EFFECT_NODE_TYPE_NOISE_REDUCTION);
    if (node.physicalNode == nullptr) {
        napi_create_int32(env, AUDIOSUITE_ERROR_SYSTEM, &ret);
        return ret;
    }

    int insertRes = -1;
    if (selectNodeId.empty()) {
        insertRes = AddEffectNodeToNodeManager(inputIdStr, uuidStr);
    } else {
        insertRes = g_nodeManager->insertNode(uuidStr, selectNodeId, Direction::LATER);
    }

    if (insertRes != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, NR_TAG, "AddEffectNodeToNodeManager ERROR!");
        napi_create_int32(env, AUDIOSUITE_ERROR_SYSTEM, &ret);
        return ret;
    }
    napi_create_int32(env, AUDIOSUITE_SUCCESS, &ret);
    return ret;
}

napi_value deleteNoiseReduction(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, NR_TAG, "audioEditTest---deleteNoiseReduction IN");
    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    std::string uuidStr;
    ParseNapiString(env, argv[NAPI_ARGV_INDEX_0], uuidStr);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, NR_TAG, "uuid==%{public}s", uuidStr.c_str());

    OH_AudioSuite_Result result;
    napi_value napiValue = nullptr;
    result = g_nodeManager->removeNode(uuidStr);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, NR_TAG, "removeNode ERROR:%{public}d", result);
    }
    napi_create_int64(env, result, &napiValue);
    return napiValue;
}