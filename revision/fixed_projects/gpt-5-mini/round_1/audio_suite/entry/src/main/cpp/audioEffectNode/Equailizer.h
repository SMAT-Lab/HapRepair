/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef AUDIOEDITTESTAPP_EQUAILIZER_H
#define AUDIOEDITTESTAPP_EQUAILIZER_H

#include <string>
#include "napi/native_api.h"
#include "ohaudio/native_audio_suite_base.h"
#include "../NodeManager.h"
#include "./EffectNode.h"

enum {
    EQ_DEFAULT = 1,
    EQ_BALLADS = 2,
    EQ_CHINESE_STYLE = 3,
    EQ_CLASSICAL = 4,
    EQ_DANCE_MUSIC = 5,
    EQ_JAZZ = 6,
    EQ_POP = 7,
    EQ_RB = 8,
    EQ_ROCK = 9,
};

struct EqBandGainsParams {
    std::string equailizerId;
    std::string inputId;
    std::string selectedNodeId;
};

OH_EqualizerFrequencyBandGains SetEqualizerMode(int32_t equailizerMode);

napi_status GetEqModeParameters(napi_env env, napi_value *argv, unsigned int &equailizerMode,
    std::string &equailizerId, std::string &inputId);

napi_status GetEqBandGainsParameters(napi_env env, napi_value *argv,
    OH_EqualizerFrequencyBandGains &frequencyBandGains, EqBandGainsParams &params);

Node GetOrCreateEqualizerNodeByMode(std::string& equailizerId, std::string& inputId);

Node GetOrCreateEqualizerNodeByGains(std::string& equailizerId, std::string& inputId, std::string& selectedNodeId);

#endif //AUDIOEDITTESTAPP_EQUAILIZER_H