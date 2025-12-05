/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "AudioSuiteError.h"

// 错误信息映射表
const std::map<AudioSuiteResult, std::string> errorMessages = {
    {AudioSuiteResult::AUDIOSUITE_SUCCESS,
        "successful"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_INVALID_PARAM,
        "This means that the function was executed with an invalid input parameter"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_ILLEGAL_STATE,
        "Execution status exception"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_SYSTEM,
        "An system error has occurred"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_UNSUPPORTED_FORMAT,
        "Unsupported audio format, such as unsupported encoding type, sample format etc"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_ENGINE_NOT_EXIST,
        "audio engine not exist"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_PIPELINE_NOT_EXIST,
        "audio pipeline not exist"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_NODE_NOT_EXIST,
        "Other Error"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_UNSUPPORTED_CONNECT,
        "Parameter Error"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_UNSUPPORT_OPERATION,
        "Remote Procedure Call Error"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_CREATED_EXCEED_SYSTEM_LIMITS,
        "System Error"},
    {AudioSuiteResult::AUDIOSUITE_ERROR_REQUIRED_PARAMETERS_MISSED,
        "System Error"},

    {AudioSuiteResult::DEMO_ERROR_FAILD,
        "Demo error"},
    {AudioSuiteResult::DEMO_PARAMETER_ANALYSIS_ERROR,
        "Parameter analysis error"},
    {AudioSuiteResult::DEMO_ERROR_PARSEARGUMENT_FAILED,
        "ParseArguments failed"},
    {AudioSuiteResult::DEMO_ERROR_CREATE_SOURCE_FAILED,
        "create source failed"},
    {AudioSuiteResult::DEMO_ERROR_TRACK_FORMAT_FAILED,
        "get track format failed"},
    {AudioSuiteResult::DEMO_ERROR_GETAUDIOPROPERTIES_FAILED,
        "get AudioProperties failed"},
    {AudioSuiteResult::DEMO_ERROR_DEMUXER_FAILED,
        "create demuxer failed"},
    {AudioSuiteResult::DEMO_UPDATEINPUTNODE_FAILED,
        "UpdateInputNode failed"},
    {AudioSuiteResult::DEMO_CREATE_NODE_ERROR,
        "Demo create node error"},
};

// 获取错误码对应的错误信息
std::string GetErrorMessage(AudioSuiteResult result)
{
    auto it = errorMessages.find(result);
    if (it != errorMessages.end()) {
        return it->second;
    } else {
        return "Unknown Error";
    }
}