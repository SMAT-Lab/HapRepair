/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef TEST1_PARSENAPIPARAM_H
#define TEST1_PARSENAPIPARAM_H

#include <string>
#include "napi/native_api.h"

class ParseNapiParam {
};

napi_status ParseNapiString(napi_env env, napi_value value, std::string &result);

#endif //TEST1_PARSENAPIPARAM_H