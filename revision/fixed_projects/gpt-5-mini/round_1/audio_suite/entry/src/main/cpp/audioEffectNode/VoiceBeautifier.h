/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
 
#ifndef AUDIOEDITTESTAPP_VOICEBEAUTIFIER_H
#define AUDIOEDITTESTAPP_VOICEBEAUTIFIER_H
 
#include <string>
#include "napi/native_api.h"
#include "./utils/Utils.h"
 
class VoiceBeautifier {};

struct StartVBParameters {
    std::string inputId;
    int mode = -1;
    std::string voiceBeautifierId;
    std::string selectNodeId; // 可选参数，为空表示不指定节点
};
 
int AddVBEffectNode(std::string inputId, int mode, std::string voiceBeautifierId, std::string selectNodeId);
 
int ModifyVBEffectNode(std::string inputId, int mode, std::string voiceBeautifierId);
 
napi_status getResetVBParameters(napi_env env, napi_value *argv, std::string &inputId, int &mode,
                                 std::string &voiceBeautifierId);
#endif // AUDIOEDITTESTAPP_VOICEBEAUTIFIER_H