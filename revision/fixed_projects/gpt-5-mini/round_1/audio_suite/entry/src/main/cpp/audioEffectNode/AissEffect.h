/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef AUDIOEDITTESTAPP_AISSEFFECT_H
#define AUDIOEDITTESTAPP_AISSEFFECT_H
#include "napi/native_api.h"

class AissEffect {
};

napi_value addAudioSeparation(napi_env env, napi_callback_info info);

napi_value deleteAudioSeparation(napi_env env, napi_callback_info info);

#endif //AUDIOEDITTESTAPP_AISSEFFECT_H