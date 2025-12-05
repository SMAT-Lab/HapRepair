/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "ParseNapiParam.h"

// 解析 napi 字符串参数
napi_status ParseNapiString(napi_env env, napi_value value, std::string &result)
{
    size_t size;
    napi_status status = napi_get_value_string_utf8(env, value, nullptr, 0, &size);
    if (status != napi_ok) {
        return status;
    }

    result.resize(size + 1); // 包含结尾的空字符
    status = napi_get_value_string_utf8(env, value, const_cast<char *>(result.data()), size + 1, nullptr);

    return status;
}
