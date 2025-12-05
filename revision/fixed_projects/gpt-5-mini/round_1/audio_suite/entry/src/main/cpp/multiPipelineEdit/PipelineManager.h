/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
#ifndef PIPELINEMANAGER_H
#define PIPELINEMANAGER_H
#include "NodeManager.h"
#include <map>

class MultiUserData {
public:
    // 根据pipelineId去获取对应的pipelineManager
    std::string pipelineId;
    // 根据inputId去writeDataBufferMap_获取对应的音频数据
    std::string inputId;
    // 音频总数据大小
    int32_t bufferSize;
    // 已经写入的音频数据大小
    size_t totalWriteAudioDataSize;
     // 音频是否从头开始写入
    bool isResetTotalWriteAudioDataSize;
public:
    MultiUserData();
    MultiUserData(std::string pipelineId, std::string inputId);
    ~MultiUserData();
};

class PipelineManager {
public:
    std::string pipelineId;
    OH_AudioSuitePipeline *audioSuitePipeline;
    std::shared_ptr<NodeManager> nodeManager;
    OH_AudioFormat audioFormatInput;
    OH_AudioFormat audioFormatOutput;
    char *firstAudioBuffer = nullptr;
    char *secondAudioBuffer = nullptr;
    char *inputBuffer = nullptr;
    bool multiRenderFrameFlag = false;
    size_t firstBufferSize = 0;
    size_t secondBufferSize = 0;
    size_t totalInputDataSize = 0;
    bool renderFrameFinishFlag = false;
    float inputDataProgress = 0.f;
    std::map<std::string, std::vector<uint8_t>> writeDataBufferMap;
    std::map<std::string, std::shared_ptr<MultiUserData>> userDataMap;
public:
    PipelineManager(std::string pipelineId, OH_AudioSuitePipeline *audioSuitePipeLine,
        std::shared_ptr<NodeManager> nodeManager);
    ~PipelineManager();
};

#endif