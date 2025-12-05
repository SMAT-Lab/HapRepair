/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
#ifndef AUDIOEDITTESTAPP_NOISEREDUCTION_H
#define AUDIOEDITTESTAPP_NOISEREDUCTION_H
#include "napi/native_api.h"

class NoiseReduction {
};

napi_value addNoiseReduction(napi_env env, napi_callback_info info);

napi_value deleteNoiseReduction(napi_env env, napi_callback_info info);
#endif //AUDIOEDITTESTAPP_NOISEREDUCTION_H