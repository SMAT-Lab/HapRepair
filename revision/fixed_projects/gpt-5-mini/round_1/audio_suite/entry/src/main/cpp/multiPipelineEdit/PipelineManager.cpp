/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
#include "PipelineManager.h"
#include <hilog/log.h>

static const int GLOBAL_RESMGR = 0xFF00;
static const char *TAG = "[AudioEditTestApp_pipelineManager_cpp]";

PipelineManager::PipelineManager(std::string pipelineId,
    OH_AudioSuitePipeline *audioSuitePipeLine, std::shared_ptr<NodeManager> nodeManager)
    : pipelineId(pipelineId), audioSuitePipeline(audioSuitePipeLine), nodeManager(nodeManager)
    {
    // 初始化PipelineManager
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG,
        "PipelineManager inited pipelineId: %{public}s, "
        "audioSuitePipeLine: %{public}p, audioSuitePipeLine: %{public}p",
        pipelineId.c_str(), audioSuitePipeLine, nodeManager.get());
}

PipelineManager::~PipelineManager()
{
}

MultiUserData::MultiUserData(std::string pipelineId, std::string inputId) : pipelineId(pipelineId), inputId(inputId)
{
}
MultiUserData::MultiUserData()
{
}
MultiUserData::~MultiUserData()
{
}