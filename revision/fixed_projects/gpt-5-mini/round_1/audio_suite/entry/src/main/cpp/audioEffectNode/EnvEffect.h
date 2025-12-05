/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef AUDIOEDITTESTAPP_ENVEFFECT_H
#define AUDIOEDITTESTAPP_ENVEFFECT_H
#include "NodeManager.h"
#include "napi/native_api.h"
#include <ohaudio/native_audio_suite_base.h>
#include <string>

struct EnvEffectParams {
    std::string inputIdStr;
    std::string uuidStr;
    unsigned int mode = 0;
    std::string selectedNodeId;
};

class EnvEffect {
};

napi_value startEnvEffect(napi_env env, napi_callback_info info);

napi_value resetEnvEffect(napi_env env, napi_callback_info info);

OH_AudioSuite_Result createEnvNodeAndSetType(std::string uuidStr, unsigned int mode,  Node &node);

#endif //AUDIOEDITTESTAPP_ENVEFFECT_H