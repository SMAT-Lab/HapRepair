/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
 
#include "VoiceBeautifier.h"
#include <string>
#include "hilog/log.h"
#include "ohaudio/native_audio_suite_base.h"
#include "ohaudio/native_audio_suite_engine.h"
#include "NodeManager.h"
#include "audioSuiteError/AudioSuiteError.h"
#include "audioEffectNode/Equailizer.h"
#include "audioEffectNode/EffectNode.h"
#include "audioEffectNode/Input.h"
#include "audioEffectNode/Output.h"
#include "realTimePlay/RealTimePlaying.h"
#include "multiPipelineEdit/MultiPipelineEdit.h"
#include "utils/Utils.h"
 
const int GLOBAL_RESMGR = 0xFF00;
const char *VB_NODE_TAG = "[AudioEditTestApp_VoiceBeautifierNode_cpp]";

int AddVBEffectNode(std::string inputId, int mode, std::string voiceBeautifierId, std::string selectNodeId)
{
    static constexpr OH_VoiceBeautifierType TYPE_MAP[] = {
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CLEAR,
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_THEATRE,
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CD,
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_RECORDING_STUDIO
    };
    OH_VoiceBeautifierType type = (mode < sizeof(TYPE_MAP) / sizeof(TYPE_MAP[0])) ? TYPE_MAP[mode] : TYPE_MAP[0];
    Node node = CreateNodeByType(voiceBeautifierId, OH_AudioNode_Type::EFFECT_NODE_TYPE_VOICE_BEAUTIFIER);
    bool bypass = mode == 0;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_BypassEffectNode(node.physicalNode, bypass);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, VB_NODE_TAG,
            "audioEditTest---startVBEffect OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        return result;
    }
    if (bypass) {
        return result;
    }
    result = OH_AudioSuiteEngine_SetVoiceBeautifierType(node.physicalNode, type);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, VB_NODE_TAG,
                     "audioEditTest---startVBEffect OH_AudioSuiteEngine_SetVoiceBeautifierType ERROR!");
        return result;
    }
    int res = -1;
    if (selectNodeId.empty()) {
        res = AddEffectNodeToNodeManager(inputId, voiceBeautifierId);
    } else {
        res = g_nodeManager->insertNode(voiceBeautifierId, selectNodeId, Direction::LATER);
    }
    if (res != 0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, VB_NODE_TAG,
                     "audioEditTest---startVBEffect AddEffectNodeToNodeManager ERROR!");
        return res;
    }
 
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, VB_NODE_TAG, "audioEditTest---startVBEffect: operation success");
    return result;
}
 
int ModifyVBEffectNode(std::string inputId, int mode, std::string voiceBeautifierId)
{
    static constexpr OH_VoiceBeautifierType TYPE_MAP[] = {
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CLEAR,
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_THEATRE,
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CD,
        OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_RECORDING_STUDIO
    };
    OH_VoiceBeautifierType type = (mode < sizeof(TYPE_MAP) / sizeof(TYPE_MAP[0])) ? TYPE_MAP[mode] : TYPE_MAP[0];
 
    Node node = g_nodeManager->GetNodeById(voiceBeautifierId);
    bool bypass = mode == 0;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_BypassEffectNode(node.physicalNode, bypass);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, VB_NODE_TAG,
            "audioEditTest---ModifyVBEffectNode OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        return result;
    }
    if (bypass) {
        return result;
    }
    result = OH_AudioSuiteEngine_SetVoiceBeautifierType(node.physicalNode, type);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, VB_NODE_TAG,
                     "audioEditTest---OH_AudioSuiteEngine_SetVoiceBeautifierType ERROR---%{public}zd", result);
        return result;
    }
 
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, VB_NODE_TAG, "audioEditTest---resetVBEffect: operation success");
    return result;
}
 
napi_status getResetVBParameters(napi_env env, napi_value *argv, std::string &inputId, int &mode,
                                 std::string &voiceBeautifierId)
{
    napi_status status = ParseNapiString(env, argv[ARG_0], inputId);
    status = napi_get_value_int32(env, argv[ARG_1], &mode);
    status = ParseNapiString(env, argv[ARG_2], voiceBeautifierId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, VB_NODE_TAG,
                 "audioEditTest resetVBEffect inputId: %{public}s, mode: %{public}d, voiceBeautifierId: %{public}s",
                 inputId.c_str(), mode, voiceBeautifierId.c_str());
    return status;
}