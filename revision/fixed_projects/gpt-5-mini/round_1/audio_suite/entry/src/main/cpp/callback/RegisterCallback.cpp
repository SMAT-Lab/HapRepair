/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "RegisterCallback.h"
#include "hilog/log.h"

const int GLOBAL_RESMGR = 0xFF00;
const char *REGISTERCALLBACK_TAG = "[AudioEditTestApp_RegisterCallback_cpp]";

napi_threadsafe_function tsfnStringArray = nullptr;
napi_threadsafe_function tsfnBoolean = nullptr;

void CallStringArrayThread(napi_env env, napi_value js_callback, void *context, void *data)
{
    std::vector<std::string> *strings = static_cast<std::vector<std::string> *>(data);
    size_t count = strings->size();
    napi_value resultArray;
    napi_create_array_with_length(env, count, &resultArray);
    for (size_t i = 0; i < count; ++i) {
        napi_value stringValue;
        napi_create_string_utf8(env, (*strings)[i].c_str(), NAPI_AUTO_LENGTH, &stringValue);
        napi_set_element(env, resultArray, i, stringValue);
    }

    delete strings; // 释放字符串数组

    napi_call_function(env, NULL, js_callback, 1, &resultArray, NULL);
}

void CallStringArrayCallback(const std::vector<std::string>& strings)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REGISTERCALLBACK_TAG,
        "audioEditTest CallStringArrayCallback called");
    if (tsfnStringArray == nullptr) {
        return;
    }
    // 创建一个新的副本
    std::vector<std::string> *stringsCopy = new std::vector<std::string>(strings);
    napi_call_threadsafe_function(tsfnStringArray, stringsCopy, napi_tsfn_blocking);
}

void CallBoolThread(napi_env env, napi_value js_callback, void *context, void *data)
{
    int result = *(bool *)data;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REGISTERCALLBACK_TAG,
        "audioEditTest CallBoolThread result: %{public}d", result);
    napi_value resultValue;
    napi_get_boolean(env, result, &resultValue);
    napi_call_function(env, NULL, js_callback, 1, &resultValue, NULL);
    free(data);
}

void CallBooleanCallback(int result)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, REGISTERCALLBACK_TAG,
        "audioEditTest CallBooleanCallback result: %{public}d", result);
    if (tsfnBoolean == nullptr) {
        return;
    }

    int *data = (int *)malloc(sizeof(int));
    if (data == nullptr) {
        return;
    }
    *data = result;

    napi_call_threadsafe_function(tsfnBoolean, data, napi_tsfn_blocking);
}