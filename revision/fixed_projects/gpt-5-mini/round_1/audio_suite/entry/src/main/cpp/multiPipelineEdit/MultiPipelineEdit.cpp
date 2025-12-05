/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <string>
#include <map>
#include <thread>
#include <algorithm>
#include <unistd.h>
#include "hilog/log.h"
#include <iomanip>
#include <fstream>
#include <fcntl.h>
#include <multiPipelineEdit/MultiPipelineEdit.h>
#include "./callback/RegisterCallback.h"
#include "./audioEffectNode/Output.h"
#include "/audioEffectNode/Equailizer.h"
#include "/audioEffectNode/VoiceBeautifier.h"
#include "/audioEffectNode/Input.h"
#include "audioEffectNode/SoundField.h"

const int GLOBAL_RESMGR = 0xFF00;
const int FIRST_ARGV_PARAM = 0;
const int SECOND_ARGV_PARAM = 1;
const int THIRD_ARGV_PARAM = 2;
const int FORTH_ARGV_PARAM = 3;
const unsigned int BITS_PER_SAMPLE_0 = 0;
const unsigned int BITS_PER_SAMPLE_1 = 1;
const unsigned int BITS_PER_SAMPLE_2 = 2;
const unsigned int BITS_PER_SAMPLE_4 = 4;
const unsigned int BITS_PER_SAMPLE_8 = 8;
const unsigned int BITS_PER_SAMPLE_16 = 16;
const unsigned int BITS_PER_SAMPLE_24 = 24;
const unsigned int BITS_PER_SAMPLE_32 = 32;
const unsigned int VB_MODE_CLEAR = 1;
const unsigned int VB_MODE_THEATRE = 2;
const unsigned int VB_MODE_CD = 3;
const unsigned int VB_MODE_RECORDING_STUDIO = 4;
const size_t INPUT_NODE_SIZE_2 = 2;
const size_t MAX_FRAME_SIZE = 20 * 192000 * 2 / 1000 * 32 / 8;
const int MAX_BUFFER_SIZE = 100 * 1024 * 1024;
const int ARRAY_SIZE_2 = 2;
const int AUDIOSUITE_ERROR_SYSTEM_CODE = 3;
const int ERROR_CODE_3 = 3;
const double HUNDRED_NUM = 100;
const char *MULTI_PIPELINE_TAG = "[AudioEditTestApp_multiPipelineEdit_cpp]";

// 多线程共享锁
std::mutex g_threadLock;
// 线程私有pipelineManager
thread_local std::shared_ptr<PipelineManager> threadPipelineManager;
// pipeline最大并行数设置为10
int g_maxPipelineSize = 10;
OH_AudioSuitePipeline **g_multiAudioSuitePipeline =
    (OH_AudioSuitePipeline **)malloc(g_maxPipelineSize * sizeof(OH_AudioSuitePipeline *));
int g_initedPipelineNum = 0;
// engine全局唯一,不可重复创建
bool g_engineInitedFlag = false;
OH_AudioSuiteEngine *g_multiAudioSuiteEngine;
// 创造 output builder 构造器
OH_AudioNodeBuilder *g_multiBuilderOut;
std::unordered_map<std::string, std::shared_ptr<PipelineManager>> pipelineIdToPipelineManagerMap;

void MultiStoreTotalBuffToMap(const char *totalBuff, size_t size, const std::string &key)
{
    std::lock_guard<std::mutex> lock(g_threadLock);
    OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest storeTotalBuffToMap totalBuff:%{public}p, size:%{public}zu", totalBuff, size);
    if (size > 0 && totalBuff != nullptr) {
        std::vector<uint8_t> buffer(totalBuff, totalBuff + size);
        threadPipelineManager->writeDataBufferMap[key] = buffer;
        return;
    }
    OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest storeTotalBuffToMap failed");
}

OH_AudioSuite_Result GetMultiRenderFrameOutput(char *&firData, char *&secData, size_t &firDataSize,
    size_t &secDataSize, bool &finishedFlag)
{
    OH_AudioSuitePipeline *threadPipeline = threadPipelineManager->audioSuitePipeline;
    OH_AudioFormat threadAudioFormatOutput = threadPipelineManager->audioFormatOutput;
    int32_t writeSize = 0;
    int32_t bitsPerSample = GetBitsPerSample(threadAudioFormatOutput.sampleFormat);
    int32_t frameSize =
        20 * threadAudioFormatOutput.samplingRate * threadAudioFormatOutput.channelCount / 1000 * bitsPerSample / 8;
    OH_AudioDataArray *ohAudioDataArray = new OH_AudioDataArray();
    ohAudioDataArray->audioDataArray = (void **)malloc(sizeof(void *) + sizeof(void *));
    for (int i = 0; i < ARRAY_SIZE_2; i++) {
        ohAudioDataArray->audioDataArray[i] = (void *)malloc(frameSize);
    }
    ohAudioDataArray->arraySize = ARRAY_SIZE_2;
    ohAudioDataArray->requestFrameSize = frameSize;
    OH_AudioSuite_Result result;

    do {
        result =
            OH_AudioSuiteEngine_MultiRenderFrame(threadPipeline, ohAudioDataArray, &writeSize, &finishedFlag);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest OH_AudioSuiteEngine_MultiRenderFrame "
                     "frameSize: %{public}d,writeSize:%{public}d,finishedFlag : %{public}s",
                     ohAudioDataArray->requestFrameSize, writeSize, (finishedFlag ? "true" : "false"));
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                         "audioEditTest OH_audioSuiteEngine_RenderFrame result is %{public}d",
                         static_cast<int>(result));
            return result;
        }

        // 每次保存一次获取的buffer值 ...
        std::copy(static_cast<const char*>(ohAudioDataArray->audioDataArray[0]),
            static_cast<const char*>(ohAudioDataArray->audioDataArray[0]) + writeSize,
            static_cast<char*>(firData) + firDataSize);
        std::copy(static_cast<const char*>(ohAudioDataArray->audioDataArray[1]),
            static_cast<const char*>(ohAudioDataArray->audioDataArray[1]) + writeSize,
            static_cast<char*>(secData) + secDataSize);
        firDataSize += writeSize;
        secDataSize += writeSize;
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest OH_AudioSuiteEngine_RenderFrame resultTotalSize: %{public}zu, writeSize : "
                     "%{public}d, finished: %{public}s",
                     firDataSize, writeSize, (finishedFlag ? "true" : "false"));
    } while (!finishedFlag);
    return result;
}

OH_AudioSuite_Result GetRenderFrameOutput(char *&firData, size_t frameSize, size_t &firDataSize, bool &finishedFlag)
{
    OH_AudioSuitePipeline *threadPipeline = threadPipelineManager->audioSuitePipeline;
    OH_AudioSuite_Result result;
    int32_t writeSize = 0;
    if (frameSize <= 0 || frameSize > MAX_FRAME_SIZE) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
            "audioEditTest GetRenderFrameOutput frameSize is invalid, cannot allocate memory");
        return AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    char *audioData = (char *)malloc(frameSize);
    // 获取管线状态
    OH_AudioSuite_PipelineState pipeLineState;
    result = OH_AudioSuiteEngine_GetPipelineState(threadPipeline, &pipeLineState);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest OH_audioSuiteEngine_GetPipelineState11111 result: %{public}d --- pipeLineState: %{public}d",
        static_cast<int>(result), static_cast<int>(pipeLineState));
    OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest renDerFrame frameSize:%{public}d", frameSize);
    do {
        result = OH_AudioSuiteEngine_RenderFrame(threadPipeline, audioData, frameSize, &writeSize, &finishedFlag);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest OH_AudioSuiteEngine_RenderFrame frameSize: %{public}zu,writeSize:%{public}d "
                     "finishedFlag : %{public}s, result: %{public}d",
                     frameSize, writeSize, (finishedFlag ? "true" : "false"), static_cast<int>(result));
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                         "audioEditTest OH_audioSuiteEngine_RenderFrame result is %{public}d",
                         static_cast<int>(result));
            break;
        }
        // 每次保存一次获取的buffer值 ...
        std::copy(static_cast<const char*>(audioData),
            static_cast<const char*>(audioData) + writeSize,
            static_cast<char*>(firData) + firDataSize);
        firDataSize += writeSize;
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest OH_AudioSuiteEngine_RenderFrame resultTotalSize: %{public}zu, writeSize : "
                     "%{public}d, finished: %{public}s",
                     firDataSize, writeSize, (finishedFlag ? "true" : "false"));
    } while (!finishedFlag);
    return result;
}

OH_AudioSuite_Result MultiPipelineRenderFrame()
{
    OH_AudioSuitePipeline *threadPipeline = threadPipelineManager->audioSuitePipeline;
    bool &multiRenderFrameFlag = threadPipelineManager->multiRenderFrameFlag;
    char *&firstAudioBuffer = threadPipelineManager->firstAudioBuffer;
    char *&secondAudioBuffer = threadPipelineManager->secondAudioBuffer;
    size_t &firstBufferSize = threadPipelineManager->firstBufferSize;
    size_t &secondBufferSize = threadPipelineManager->secondBufferSize;
    OH_AudioFormat threadAudioFormatOutput = threadPipelineManager->audioFormatOutput;
    bool &finishedFlag = threadPipelineManager->renderFrameFinishFlag;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,"audioEditTest RenDerFrame start, "
                 "pipeline:%{public}p", threadPipeline);

    OH_AudioSuite_Result result = StartPipelineAndCheckState();
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "StartPipeline ERROR:%{public}d", result);
        return result;
    }
    
    char *firAudioData = (char *)malloc(1024 * 1024 * 100);
    char *secAudioData = (char *)malloc(1024 * 1024 * 100);
    
    int32_t bitsPerSample = GetBitsPerSample(threadAudioFormatOutput.sampleFormat);
    int32_t frameSize =
        20 * threadAudioFormatOutput.samplingRate * threadAudioFormatOutput.channelCount / 1000 * bitsPerSample / 8;
    
    if (multiRenderFrameFlag) {
        result = GetMultiRenderFrameOutput(firAudioData, secAudioData, firstBufferSize, secondBufferSize, finishedFlag);
    } else {
        result = GetRenderFrameOutput(firAudioData, frameSize, firstBufferSize, finishedFlag);
    }
    if (result == OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        FreeBuffer(firstAudioBuffer);
        firstAudioBuffer = (char *)malloc(firstBufferSize);
        std::copy(static_cast<const char*>(firAudioData),
            static_cast<const char*>(firAudioData) + firstBufferSize,
            static_cast<char*>(firstAudioBuffer));
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest renDerFrame std::copy "
                     "firBuff: %{public}p, firstBufferSize:%{public}zu", firstAudioBuffer, firstBufferSize);
        if (multiRenderFrameFlag) {
            FreeBuffer(secondAudioBuffer);
            secondAudioBuffer = (char *)malloc(secondBufferSize);
            std::copy(static_cast<const char*>(secAudioData),
                static_cast<const char*>(secAudioData) + secondBufferSize,
                static_cast<char*>(secondAudioBuffer));
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest renDerFrame "
                         "std::copy secBuff: %{public}p, g_totalSize:%{public}zu",
                         secondAudioBuffer, secondBufferSize);
            multiRenderFrameFlag = false;
        }
    }
    return result;
}

napi_value MultiPipelineEnvPrepare(napi_env env, napi_callback_info info)
{
    std::lock_guard<std::mutex> lock(g_threadLock);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest multiPipelinePrepare start");
    napi_value napiValue;
    OH_AudioSuite_Result result;

    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    std::string pipelineId;
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    ParseNapiString(env, argv[FIRST_ARGV_PARAM], pipelineId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest multiPipelinePrepare pipelineId=%{public}s", pipelineId.c_str());

    if (threadPipelineManager != nullptr && threadPipelineManager->pipelineId == pipelineId) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "multiPipelinePrepare direct return, pipeline manager of thread had inited,"
                     "threadPipelineManager=%{public}p",
                     threadPipelineManager.get());
        napi_create_int64(env, static_cast<int>(0), &napiValue);
        return napiValue;
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "multiPipelinePrepare thread_id=%{public}d",
                 gettid());
    std::shared_ptr<PipelineManager> pipelineManager = pipelineIdToPipelineManagerMap[pipelineId];
    if (pipelineManager == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "multiPipelinePrepare pipelineIdToPipelineManagerMap ERROR pipelineId=%{public}s",
                     pipelineId.c_str());
    }
    threadPipelineManager = pipelineManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "multiPipelinePrepare threadPipelineManager=%{public}p", threadPipelineManager.get());
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

napi_value AudioEditNodeInitMultiPipeline(napi_env env, napi_callback_info info)
{
    std::lock_guard<std::mutex> lock(g_threadLock);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest AudioEditNodeInitMultiPipeline start");
    OH_AudioSuite_Result result;
    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    std::string pipelineId;
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    ParseNapiString(env, argv[FIRST_ARGV_PARAM], pipelineId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest AudioEditNodeInitMultiPipeline pipelineId:%{public}s", pipelineId.c_str());

    // 创建引擎
    if (!g_engineInitedFlag) {
        g_engineInitedFlag = true;
        result = OH_AudioSuiteEngine_Create(&g_multiAudioSuiteEngine);
    }
    OH_AudioSuite_PipelineWorkMode workMode;

    // 创建管线
    result = OH_AudioSuiteEngine_CreatePipeline(g_multiAudioSuiteEngine,
        &g_multiAudioSuitePipeline[g_initedPipelineNum], workMode);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest OH_AudioEditEngine_CreatePipeline result: %{public}d", static_cast<int>(result));
    // 实例化PipelineManager
    std::shared_ptr<NodeManager> nodeManager =
        std::make_shared<NodeManager>(g_multiAudioSuitePipeline[g_initedPipelineNum]);
    std::shared_ptr<PipelineManager> pipelineManager =
        std::make_shared<PipelineManager>(pipelineId, g_multiAudioSuitePipeline[g_initedPipelineNum], nodeManager);
    pipelineIdToPipelineManagerMap[pipelineId] = pipelineManager;
    OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest multi pipeline init: g_initedPipelineNum: %{public}d,"
                 "pipelineManager:%{public}p, audioSuitePipeline: %{public}p, nodeManager:%{public}p",
                 g_initedPipelineNum, pipelineManager.get(), pipelineManager->audioSuitePipeline,
                 pipelineManager->nodeManager.get());
    
    for (const auto &pair : pipelineIdToPipelineManagerMap) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "pipelineIdToPipelineManagerMap key=%{public}s,value=%{public}p", pair.first.c_str(),
                     pair.second.get());
    }
    g_initedPipelineNum++;
    napi_value napiValue;
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

napi_value DestroyMultiPipeline(napi_env env, napi_callback_info info)
{
    std::lock_guard<std::mutex> lock(g_threadLock);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest AudioEditDestory start");
    OH_AudioSuite_Result result = AUDIOSUITE_SUCCESS;
    for (const auto &pair : pipelineIdToPipelineManagerMap) {
        result = OH_AudioSuiteEngine_DestroyPipeline(pair.second->audioSuitePipeline);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest OH_audioSuiteEngine_DestroyPipeline result: %{public}d, pipeline:%{public}p",
                     static_cast<int>(result), pair.second->audioSuitePipeline);
        if (result != 0) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "destoryMultiPipeline ERROR");
        }
    }
    g_initedPipelineNum = 0;
    pipelineIdToPipelineManagerMap.clear();
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest clear all map of multi pipeline");

    result = OH_AudioSuiteEngine_Destroy(g_multiAudioSuiteEngine);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest OH_audioSuiteEngine_Destroy result: %{public}d", static_cast<int>(result));
    g_engineInitedFlag = false;
    napi_value napiValue;
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

napi_value MultiSetFormat(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest SetFormat start");
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // 获取通道数
    unsigned int channels;
    unsigned int sampleRate;
    unsigned int bitsPerSample;
    napi_get_value_uint32(env, argv[FIRST_ARGV_PARAM], &channels);
    napi_get_value_uint32(env, argv[SECOND_ARGV_PARAM], &sampleRate);
    napi_get_value_uint32(env, argv[THIRD_ARGV_PARAM], &bitsPerSample);
    switch (bitsPerSample) {
        case BITS_PER_SAMPLE_8:
            bitsPerSample = BITS_PER_SAMPLE_0;
            break;
        case BITS_PER_SAMPLE_16:
            bitsPerSample = BITS_PER_SAMPLE_1;
            break;
        case BITS_PER_SAMPLE_24:
            bitsPerSample = BITS_PER_SAMPLE_2;
            break;
        case BITS_PER_SAMPLE_32:
            bitsPerSample = BITS_PER_SAMPLE_4;
            break;
        default:
            bitsPerSample = BITS_PER_SAMPLE_0;
            break;
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest SetFormat channels: %{public}d, sampleRate: %{public}d, bitsPerSample: %{public}d",
                 channels, sampleRate, bitsPerSample);
    OH_AudioFormat &threadAudioFormatOutput = threadPipelineManager->audioFormatOutput;
    const std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    // 设置采样率
    threadAudioFormatOutput.samplingRate = SetSamplingRate(sampleRate);
    // 设置声道
    threadAudioFormatOutput.channelCount = channels;
    threadAudioFormatOutput.channelLayout = SetChannelLayout(channels);
    // 设置位深
    threadAudioFormatOutput.sampleFormat = SetSampleFormat(bitsPerSample);
    // 设置编码格式
    threadAudioFormatOutput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest SetFormat threadAudioFormatOutput is %{public}p", &threadAudioFormatOutput);
    const std::vector<Node> outPutNodes =
        threadNodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    OH_AudioSuite_Result result =
        OH_AudioSuiteEngine_SetAudioFormat(outPutNodes[0].physicalNode, &threadAudioFormatOutput);
    napi_value napiValue;
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

int32_t MultiWriteDataCallBack(OH_AudioNode *audioNode, void *userData, void *audioData, int32_t audioDataSize,
                               bool *finished)
{
    std::lock_guard<std::mutex> lock(g_threadLock);
    // 检查audioNode参数， 底层接口问题
    if (audioNode == nullptr || audioData == nullptr || finished == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "WriteDataCallBack audioNode is nullptr");
        *finished = true;
        return 0;
    }
    OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "multiWriteDataCallBack start");
    
    // 处理音频数据   此处如果是nullptr，是demo获取音频数据的问题，非底层接口问题
    MultiUserData *curMultiUserData = static_cast<MultiUserData *>(userData);
    std::string inputId = curMultiUserData->inputId;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,"WriteDataCallBack inputId: %{public}s,"
                  "pipelineId:%{public}s", inputId.c_str(), curMultiUserData->pipelineId.c_str());
    std::shared_ptr<PipelineManager> threadPipelineManager =
        pipelineIdToPipelineManagerMap[curMultiUserData->pipelineId];
    std::map<std::string, std::vector<uint8_t>> &writeDataBufferMap = threadPipelineManager->writeDataBufferMap;
    float &inputDataProgress = threadPipelineManager->inputDataProgress;
    int32_t totalSize = curMultiUserData->bufferSize;
    size_t &totalWriteAudioDataSize = curMultiUserData->totalWriteAudioDataSize;
    auto it = writeDataBufferMap.find(inputId);
    if (it == writeDataBufferMap.end()) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest WriteDataCallBack writeDataBufferMap_ is end");
        *finished = true;
        return 0;
    }
    int32_t remainingDataSize = totalSize - totalWriteAudioDataSize;
    int32_t actualDataSize = std::min(audioDataSize, remainingDataSize);
    inputDataProgress += (static_cast<double>(actualDataSize) / static_cast<double>(totalSize) * HUNDRED_NUM);
    // 将数据从inputBuffer复制到audioData
    std::copy(reinterpret_cast<const char*>(it->second.data() + totalWriteAudioDataSize),
        reinterpret_cast<const char*>(it->second.data() + totalWriteAudioDataSize) + actualDataSize,
        static_cast<char*>(audioData));
    // 跟新已写入的数据量
    totalWriteAudioDataSize += actualDataSize;
    int32_t padSize = audioDataSize - remainingDataSize;
    if (padSize > 0) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "padSize: %{public}d", padSize);
        std::fill(static_cast<char*>(audioData) + actualDataSize,
            static_cast<char*>(audioData) + actualDataSize + padSize, 0);
    }

    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest WriteDataCallBack totalSize: %{public}d, totalWriteAudioDataSize: %{public}zu, "
                 "audioDataSize: %{public}d, actualDataSize:%{public}d, padSize: %{public}d",
                 totalSize, totalWriteAudioDataSize, audioDataSize, actualDataSize, padSize);
    if (totalWriteAudioDataSize >= totalSize) {
        inputDataProgress = 100.f;
    }
    return actualDataSize;
}

OH_AudioSuite_Result MultiSetParamsAndWriteData(OH_AudioNodeBuilder *builder, std::string inputId,
                                                OH_AudioNode_Type type)
{
    OH_AudioSuite_Result result = OH_AudioSuiteNodeBuilder_SetFormat(builder, threadPipelineManager->audioFormatInput);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest OH_AudioNodeBuilder_SetFormat result is %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    if (type != OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT) {
        return result;
    }
    std::shared_ptr<MultiUserData> userData =
        std::make_shared<MultiUserData>(threadPipelineManager->pipelineId, inputId);
    // 后面可以考虑去掉totalInputDataSize，用入参形式传入
    userData->inputId = inputId;
    userData->pipelineId = threadPipelineManager->pipelineId;
    userData->bufferSize = threadPipelineManager->totalInputDataSize;
    userData->totalWriteAudioDataSize = 0;
    userData->isResetTotalWriteAudioDataSize = false;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest OH_AudioSuiteNodeBuilder_SetRequestDataCallback"
                 "data address is %{public}p",
                 userData.get());
    OH_LOG_Print(
        LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest OH_AudioNodeBuilder_SetFormat MultiUserData inputId is %{public}s, pipelineId is %{public}s",
        userData->inputId.c_str(), userData->pipelineId.c_str());
    // 设置OH_AudioSuiteNodeBuilder_SetRequestDataCallback回调, 创建节点之前
    result = OH_AudioSuiteNodeBuilder_SetRequestDataCallback(builder, MultiWriteDataCallBack, (void *)userData.get());
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest OH_AudioSuiteNodeBuilder_SetRequestDataCallback result is %{public}d",
                 static_cast<int>(result));
    
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    // 将MultiUserData实例存入映射表中
    threadPipelineManager->userDataMap[inputId] = userData;
    return result;
}

void MultiCreateInputNode(napi_env env, const std::string &inputId, napi_value &napiValue,
                          OH_AudioSuite_Result &result)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest createInputNode start");
    // 添加音频，将音频的buffer出存储到map中，， 上一行中的memcpy可以考虑删除了
    char *threadinputBuffer = threadPipelineManager->inputBuffer;
    size_t threadtotalInputDataSize = threadPipelineManager->totalInputDataSize;
    MultiStoreTotalBuffToMap(threadinputBuffer, threadtotalInputDataSize, inputId);
    auto it = threadPipelineManager->writeDataBufferMap.find(inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest AudioInAndOutInit writeDataBufferMap_[inputId] length: %{public}zu", it->second.size());
    // 创造 builder 构造器
    OH_AudioNodeBuilder *builderIn;
    result = OH_AudioSuiteNodeBuilder_Create(&builderIn);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest OH_AudioSuiteNodeBuilder_Create result: %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        return;
    }
    // 透传节点类型
    result = OH_AudioSuiteNodeBuilder_SetNodeType(builderIn, OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "NodeManagerTest createNode OH_AudioSuiteNodeBuilder_SetNodeType result: %{public}d",
                 static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        return;
    }

    // 封装方法，设置 音频文件的 参数 以及 写入音频文件到缓冲区
    result = MultiSetParamsAndWriteData(builderIn, inputId, OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest SetParamsAndWriteData result: %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        return;
    }

    // 创建input节点
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest CreateInputNode threadNodeManager: %{public}p , inputId:%{public}s",
                 threadNodeManager.get(), inputId.c_str());
    threadNodeManager->createNode(inputId, OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT, builderIn);
}

void MultiUpdateInputNode(OH_AudioSuite_Result &result, const UpdateInputNodeParams &params)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest MultiUpdateInputNode start");
    OH_AudioFormat &audioFormatInput = threadPipelineManager->audioFormatInput;
    OH_AudioFormat &audioFormatOutput = threadPipelineManager->audioFormatOutput;
    std::map<std::string, std::vector<uint8_t>> writeDataBufferMap = threadPipelineManager->writeDataBufferMap;
    // 设置采样率
    audioFormatInput.samplingRate = SetSamplingRate(params.sampleRate);
    // 设置声道
    audioFormatInput.channelCount = params.channels;
    audioFormatInput.channelLayout = SetChannelLayout(params.channels);
    // 设置位深
    audioFormatInput.sampleFormat = SetSampleFormat(params.bitsPerSample);
    // 设置编码格式
    audioFormatInput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;
    
    audioFormatOutput.samplingRate = audioFormatInput.samplingRate;
    audioFormatOutput.channelCount = params.channels;
    audioFormatOutput.channelLayout = audioFormatInput.channelLayout;
    audioFormatOutput.sampleFormat = audioFormatInput.sampleFormat;
    audioFormatOutput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;

    const std::vector<Node> inPutNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    result = OH_AudioSuiteEngine_SetAudioFormat(inPutNodes[0].physicalNode, &audioFormatInput);
    const std::vector<Node> outPutNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    result = OH_AudioSuiteEngine_SetAudioFormat(outPutNodes[0].physicalNode, &audioFormatOutput);
    // 添加音频，将音频的buffer出存储到map中，， 上一行中的memcpy可以考虑删除了
    if (g_writeDataBufferMap.find(params.inputId) != g_writeDataBufferMap.end()) {
        // 键存在，执行删除操作
        g_writeDataBufferMap.erase(params.inputId);
    }
    MultiStoreTotalBuffToMap(threadPipelineManager->inputBuffer, threadPipelineManager->totalInputDataSize,
                             params.inputId);
    auto it = g_writeDataBufferMap.find(params.inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest AudioInAndOutInit g_writeDataBufferMap[inputId] length: %{public}d", it->second.size());
    UserData *data = new UserData();
    data->id = params.inputId;
    // 后面可以考虑去掉totalSize_，用入参形式传入
    data->bufferSize = g_totalSize;
    data->totalWriteAudioDataSize = 0;
    data->isResetTotalWriteAudioDataSize = false;
}

void MultiReadTrackSamples(OH_AVDemuxer *demuxer, uint32_t trackIndex, int bufferSize, std::atomic<bool> &isEnd,
                           std::shared_ptr<PipelineManager> threadPipelineManager)
{
    char *&threadinputBuffer = threadPipelineManager->inputBuffer;
    size_t &threadtotalInputDataSize = threadPipelineManager->totalInputDataSize;
    threadtotalInputDataSize = 0;
    threadinputBuffer = nullptr;
    // 添加解封装轨道
    if (OH_AVDemuxer_SelectTrackByID(demuxer, trackIndex) != AV_ERR_OK) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "select audio track failed: %{pbulic}d",
                     trackIndex);
    }
    // 创建缓冲区
    if (bufferSize <= 0 || bufferSize > MAX_BUFFER_SIZE) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
            "audioEditTest GetRenderFrameOutput bufferSize is invalid, cannot allocate memory");
        return;
    }
    OH_AVBuffer *buffer = OH_AVBuffer_Create(bufferSize);
    char *totalBuffer = (char *)malloc(bufferSize);
    if (buffer == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "create buffer failed");
    }
    OH_AVCodecBufferAttr info;
    int32_t ret;

    while (!isEnd.load()) {
        ret = OH_AVDemuxer_ReadSampleBuffer(demuxer, trackIndex, buffer);
        if (ret == AV_ERR_OK) {
            OH_AVBuffer_GetBufferAttr(buffer, &info);
            // 将当前样本的数据复制到 totalBuff 中
            std::copy(reinterpret_cast<const char*>(OH_AVBuffer_GetAddr(buffer)),
                reinterpret_cast<const char*>(OH_AVBuffer_GetAddr(buffer)) + info.size,
                static_cast<char*>(totalBuffer) + threadtotalInputDataSize);
            threadtotalInputDataSize += info.size;
            if (info.flags == OH_AVCodecBufferFlags::AVCODEC_BUFFER_FLAGS_EOS) {
                isEnd.store(true);
            }
        } else {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                         "get buffer failed");
        }
    }
    threadinputBuffer = (char *)malloc(threadtotalInputDataSize);
    std::copy(static_cast<const char*>(totalBuffer),
        static_cast<const char*>(totalBuffer) + threadtotalInputDataSize,
        static_cast<char*>(threadinputBuffer));
    // 销毁缓冲区
    free(totalBuffer);
    OH_AVBuffer_Destroy(buffer);
}

bool MultiGetAudioProperties(OH_AVFormat *trackFormat, int32_t &sampleRate, int32_t &channels, int32_t &bitsPerSample)
{
    if (!OH_AVFormat_GetIntValue(trackFormat, OH_MD_KEY_AUD_SAMPLE_RATE, &sampleRate)) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "get sample rate failed");
        return false;
    }
    if (!OH_AVFormat_GetIntValue(trackFormat, OH_MD_KEY_AUD_CHANNEL_COUNT, &channels)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "get channel count failed");
        return false;
    }
    if (!OH_AVFormat_GetIntValue(trackFormat, OH_MD_KEY_AUDIO_SAMPLE_FORMAT, &bitsPerSample)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "get bits per sample failed");
        return false;
    }
    OH_AudioFormat &threadAudioFormatInput = threadPipelineManager->audioFormatInput;
    OH_AudioFormat &threadAudioFormatOutput = threadPipelineManager->audioFormatOutput;
    // 设置采样率
    threadAudioFormatInput.samplingRate = SetSamplingRate(sampleRate);
    // 设置声道
    threadAudioFormatInput.channelCount = channels;
    threadAudioFormatInput.channelLayout = SetChannelLayout(channels);
    // 设置位深
    threadAudioFormatInput.sampleFormat = SetSampleFormat(bitsPerSample);
    // 设置编码格式
    threadAudioFormatInput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;

    threadAudioFormatOutput.samplingRate = threadAudioFormatInput.samplingRate;
    threadAudioFormatOutput.channelCount = channels;
    threadAudioFormatOutput.channelLayout = threadAudioFormatInput.channelLayout;
    threadAudioFormatOutput.sampleFormat = threadAudioFormatInput.sampleFormat;
    threadAudioFormatOutput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;

    return true;
}

void MultiRunAudioThread(OH_AVDemuxer *demuxer, int32_t fileLength)
{
    std::atomic<bool> audioIsEnd{false};
    std::thread audioThread(MultiReadTrackSamples, demuxer, 0, fileLength, std::ref(audioIsEnd), threadPipelineManager);
    audioThread.join();
}

void MultiManageExistingOutputNodes(const std::string &inputId, const std::string &mixerId,
                                    OH_AudioSuite_Result &result, std::vector<Node> outPutNodes)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest AddEffectNodeToNodeManager start, threadNodeManager: %{public}p",
                 threadNodeManager.get());
    const std::vector<Node> mixerNodes =
        threadNodeManager->getNodesByType(OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER);
    if (mixerNodes.size() > 0) {
        result = threadNodeManager->connect(inputId, mixerNodes[0].id);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest connect input and mixer result: %{public}d", static_cast<int>(result));
    } else {
        result = threadNodeManager->createNode(mixerId, OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest nodeManagerCreateMixerNode result: %{public}d", static_cast<int>(result));

        result = threadNodeManager->insertNode(mixerId, outPutNodes[0].id, Direction::BEFORE);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest insertMixerNode result: %{public}d", static_cast<int>(result));

        result = threadNodeManager->connect(inputId, mixerId);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest connect inputId and mixerId result: %{public}d", static_cast<int>(result));
    }
}

void MultiCreateAndConnectOutputNodes(const std::string &inputId, const std::string &outputId,
                                      OH_AudioSuite_Result &result)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest CreateAndConnectOutputNodes start, threadNodeManager: %{public}p",
                 threadNodeManager.get());
    result = OH_AudioSuiteNodeBuilder_Create(&g_multiBuilderOut);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest CreateAndConnectOutputNodes output builder result: %{public}d",
                 static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return;
    }

    result = OH_AudioSuiteNodeBuilder_SetNodeType(g_multiBuilderOut, OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "NodeManagerTest createNode OH_AudioSuiteNodeBuilder_SetNodeType result: %{public}d",
                 static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return;
    }

    result = MultiSetParamsAndWriteData(g_multiBuilderOut, inputId, OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest SetParamsAndWriteData result: %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return;
    }

    result = threadNodeManager->createNode(outputId, OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT, g_multiBuilderOut);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest nodeManagerCreateOutputNode result: %{public}d", static_cast<int>(result));

    result = threadNodeManager->connect(inputId, outputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest nodeManagerConnectInputAndOutput result: %{public}d", static_cast<int>(result));
}

void MultiManageOutputNodes(napi_env env, const std::string &inputId, const std::string &outputId,
                            const std::string &mixerId, OH_AudioSuite_Result &result)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest ManageOutputNodes start, threadNodeManager: %{public}p", threadNodeManager.get());
    const std::vector<Node> outPutNodes =
        threadNodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    if (outPutNodes.size() > 0) {
        MultiManageExistingOutputNodes(inputId, mixerId, result, outPutNodes);
    } else {
        MultiCreateAndConnectOutputNodes(inputId, outputId, result);
    }
}

napi_value MultiAudioInAndOutInit(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest AudioInAndOutInit start");

    size_t argc = 5;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    AudioParams params;
    napi_status status = ParseArguments(env, argv, params);
    if (status != napi_ok) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(status));
    }
    OH_AVSource *source = OH_AVSource_CreateWithFD(params.fd, 0, params.fileLength);
    if (source == nullptr) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    OH_AVFormat *trackFormat = OH_AVSource_GetTrackFormat(source, 0);
    if (trackFormat == nullptr) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    int32_t sampleRate;
    int32_t channels;
    int32_t bitsPerSample;
    if (!MultiGetAudioProperties(trackFormat, sampleRate, channels, bitsPerSample)) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "sampleRate: %{public}d, channels: %{public}d,"
                 "bitsPerSample: %{public}d", sampleRate, channels, bitsPerSample);
    OH_AVDemuxer *demuxer = OH_AVDemuxer_CreateWithSource(source);
    if (demuxer == nullptr) {
        return ReturnResult(env, AudioSuiteResult::DEMO_ERROR_FAILD);
    }
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    MultiRunAudioThread(demuxer, params.fileLength);
    napi_value napiValue;
    OH_AudioSuite_Result result;
    Node inputNode = threadNodeManager->GetNodeById(params.inputId);
    if (inputNode.id.empty()) {
        MultiCreateInputNode(env, params.inputId, napiValue, result);
    } else {
        UpdateInputNodeParams params;
        MultiUpdateInputNode(result, params);
        return ReturnResult(env, static_cast<AudioSuiteResult>(result));
    }
    MultiManageOutputNodes(env, params.inputId, params.outputId, params.mixerId, result);
    std::vector<std::string> audioFormat = {std::to_string(sampleRate), std::to_string(channels),
                                            std::to_string(bitsPerSample)};
    CallStringArrayCallback(audioFormat);
    return ReturnResult(env, static_cast<AudioSuiteResult>(result));
}

int MultiAddEffectNodeToNodeManager(std::string &inputNodeId, std::string &effectNodeId)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(
        LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest AddEffectNodeToNodeManager start and inputNodeId is : %{public}s, effectNodeId is %{public}s,"
        "threadNodeManager: %{public}p",
        inputNodeId.c_str(), effectNodeId.c_str(), threadNodeManager.get());
    // 添加效果节点，检查是否有混音节点，没有混音节点就将效果节点添加到output节点之前；有混音节点，获取到对应input节点id，按序插入到混音节点之前
    const std::vector<Node> mixerNodes =
        threadNodeManager->getNodesByType(OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER);
    OH_AudioSuite_Result result;
    Node node = threadNodeManager->GetNodeById(effectNodeId);
    if (node.id.empty()) {
        return -1;
    }

    if (mixerNodes.size() > 0) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest AddEffectNodeToNodeManager has mixerNodes");
        Node node = threadNodeManager->GetNodeById(inputNodeId);
        if (node.nextNodeId.empty()) {
            return -ERROR_CODE_3;
        }
        while (threadNodeManager->GetNodeById(node.nextNodeId).type !=
               OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER) {
            node = threadNodeManager->GetNodeById(node.nextNodeId);
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                         "audioEditTest AddEffectNodeToNodeManager has mixerNodes and nextNode : %{public}s",
                         node.id.c_str());
        }
        result = threadNodeManager->insertNode(effectNodeId, node.id, Direction::LATER);
    } else {
        const std::vector<Node> outPutNodes =
            threadNodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
        result = threadNodeManager->insertNode(effectNodeId, outPutNodes[0].id, Direction::BEFORE);
    }

    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest AddEffectNodeToNodeManager end and result is: %{public}d", static_cast<int>(result));

    return result;
}

OH_AudioSuite_Result RemoveNodeMoreThanTwoSize(std::shared_ptr<NodeManager> &threadNodeManager, std::string inputId)
{
    Node node = threadNodeManager->GetNodeById(inputId);
    Node nextNode;
    if (node.id.empty()) {
        return AUDIOSUITE_ERROR_SYSTEM;
    }
    while (node.type != OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER) {
        nextNode = threadNodeManager->GetNodeById(node.nextNodeId);
        OH_AudioSuite_Result result = threadNodeManager->removeNode(node.id);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
        node = nextNode;
    }
    OH_LOG_Print(
        LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest deleteSong preNodes of mixerNode and inputNodes number greater than 2 : %{public}d",
        static_cast<int>(threadNodeManager->getNodesByType(OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER)[0]
                             .preNodeIds.size()));
    return AUDIOSUITE_SUCCESS;
}

OH_AudioSuite_Result RemoveNodeEqualTwoSize(std::shared_ptr<NodeManager> &threadNodeManager, std::string inputId)
{
    Node node = threadNodeManager->GetNodeById(inputId);
    Node nextNode;
    if (node.id.empty()) {
        return AUDIOSUITE_ERROR_SYSTEM;
    }
    while (node.type != OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT) {
        nextNode = threadNodeManager->GetNodeById(node.nextNodeId);
        OH_AudioSuite_Result result = threadNodeManager->removeNode(node.id);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
        node = nextNode;
    }
    OH_LOG_Print(
        LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest deleteSong number of mixerNode : %{public}d",
        static_cast<int>(
            threadNodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT)[0].preNodeIds.size()));
    return AUDIOSUITE_SUCCESS;
}

OH_AudioSuite_Result RemoveNodeEqualOneSize(std::shared_ptr<NodeManager> &threadNodeManager, std::string inputId)
{
    Node node = threadNodeManager->GetNodeById(inputId);
    Node nextNode;
    while (!node.id.empty()) {
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "deleteSong inputNodes is 1 : %{public}s", node.id.c_str());
    nextNode = threadNodeManager->GetNodeById(node.nextNodeId);
    OH_AudioSuite_Result result = threadNodeManager->removeNode(node.id);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    node = nextNode;
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest deleteSong nodes : %{public}zu", threadNodeManager->getAllNodes().size());
    return AUDIOSUITE_SUCCESS;
}

napi_value MultiDeleteSong(napi_env env, napi_callback_info info)
{
    OH_AudioSuite_Result result;
    napi_value napiValue;
    size_t argc = 1;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    std::string inputId;
    napi_status status = ParseNapiString(env, argv[0], inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "DeleteSong inputId:%{public}s", inputId.c_str());
    if (threadPipelineManager == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "DeleteSong, threadNodeManager is null");
        napi_create_int64(env, static_cast<int>(AUDIOSUITE_ERROR_SYSTEM), &napiValue);
        return napiValue;
    }
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    const std::vector<Node> inputNodes = threadNodeManager->getNodesByType(OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest DeleteSong inputNodes length is %{public}d", static_cast<int>(inputNodes.size()));
    if (inputNodes.size() > INPUT_NODE_SIZE_2) {
        result = RemoveNodeMoreThanTwoSize(threadNodeManager, inputId);
    } else if (inputNodes.size() == INPUT_NODE_SIZE_2) {
        result = RemoveNodeEqualTwoSize(threadNodeManager, inputId);
    } else if (inputNodes.size() == 1) {
        result = RemoveNodeEqualOneSize(threadNodeManager, inputId);
    } else {
        napi_create_int64(env, static_cast<int>(-1), &napiValue);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "deleteSong inputNodes less than 1");
        return napiValue;
    }
    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}

napi_value MultiSetEqualizerMode(napi_env env, napi_callback_info info)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "SetEquailizerMode start, threadNodeManager:%{public}p", threadNodeManager.get());
    napi_value napiValue;
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    unsigned int equailizerMode = -1;
    napi_get_value_uint32(env, argv[FIRST_ARGV_PARAM], &equailizerMode);
    std::string equalizerId;
    std::string inputId;
    napi_status status = ParseNapiString(env, argv[SECOND_ARGV_PARAM], equalizerId);
    status = ParseNapiString(env, argv[THIRD_ARGV_PARAM], inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "equalizerId: %{public}s, inputId: %{public}s, "
                 "equailizerMode:%{public}d,", equalizerId.c_str(), inputId.c_str(), equailizerMode);
    
    Node eqNode = threadNodeManager->GetNodeById(equalizerId);
    if (eqNode.physicalNode) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest SetEquailizerMode equalizer is exist");
    } else {
        // 创建均衡器节点
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest SetEquailizerMode crate EQUALIZER node");
        eqNode.id = equalizerId;
        eqNode.type = OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER;
        threadNodeManager->createNode(equalizerId, OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER);
        // 获取效果节点
        eqNode = threadNodeManager->GetNodeById(equalizerId);
        int resultInt = MultiAddEffectNodeToNodeManager(inputId, equalizerId);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest AddEffectNodeToNodeManager result: %{public}d", resultInt);
        if (resultInt != 0) {
            napi_create_int64(env, resultInt, &napiValue);
            return napiValue;
        }
    }

    OH_AudioSuite_Result result =
        OH_AudioSuiteEngine_SetEqualizerFrequencyBandGains(eqNode.physicalNode, SetEqualizerMode(equailizerMode));
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest OH_AudioSuiteEngine_SetEqualizerMode result: %{public}d", static_cast<int>(result));

    napi_create_int64(env, static_cast<int>(result), &napiValue);
    return napiValue;
}
Node MultiGetOrCreateEqualizerNodeByGains(std::string& equailizerId, std::string& inputId, std::string& selectedNodeId)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    Node eqNode = threadNodeManager->GetNodeById(equailizerId);
    if (!eqNode.physicalNode) {
        // 创建均衡器节点
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest SetEqualizerFrequencyBandGains crate equalizer node");
        eqNode.id = equailizerId;
        eqNode.type = OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER;
        threadNodeManager->createNode(equailizerId, OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER);
        // 获取效果节点
        eqNode = threadNodeManager->GetNodeById(equailizerId);
        if (selectedNodeId.empty()) {
            int resultInt = MultiAddEffectNodeToNodeManager(inputId, equailizerId);
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                         "audioEditTest AddEffectNodeToNodeManager AddEffectNodeToNodeManager result: %{public}d",
                         resultInt);
            if (resultInt != 0) {
                eqNode.physicalNode = nullptr; // 标记为失败
            }
        } else {
            int resultInt = threadNodeManager->insertNode(equailizerId, selectedNodeId, Direction::LATER);
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                         "audioEditTest AddEffectNodeToNodeManager insertNode result: %{public}d", resultInt);
            if (resultInt != 0) {
                eqNode.physicalNode = nullptr; // 标记为失败
            }
        }
    }
    return eqNode;
}

napi_value MultiSetEqualizerFrequencyBandGains(napi_env env, napi_callback_info info)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "SetEqualizerFrequencyBandGains start, threadNodeManager:%{public}p", threadNodeManager.get());
    size_t argc = 4;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    OH_EqualizerFrequencyBandGains frequencyBandGains;
    EqBandGainsParams params;
    napi_status status = GetEqBandGainsParameters(env, argv, frequencyBandGains, params);
    if (status != napi_ok) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(AudioSuiteResult::DEMO_PARAMETER_ANALYSIS_ERROR));
    }
    // 创建均衡器频带节点
    Node eqNode = MultiGetOrCreateEqualizerNodeByGains(params.equailizerId, params.inputId, params.selectedNodeId);
    if (!eqNode.physicalNode) {
        return ReturnResult(env, static_cast<AudioSuiteResult>(AudioSuiteResult::DEMO_CREATE_NODE_ERROR));
    }
    
    OH_AudioSuite_Result result =
        OH_AudioSuiteEngine_SetEqualizerFrequencyBandGains(eqNode.physicalNode, frequencyBandGains);
    return ReturnResult(env, static_cast<AudioSuiteResult>(result));
}

napi_value MultiSaveFileBuffer(napi_env env, napi_callback_info info)
{
    pthread_t thisId = gettid();
    OH_AudioSuitePipeline *threadPipeline = threadPipelineManager->audioSuitePipeline;
    char *&threadBuffer = threadPipelineManager->firstAudioBuffer;
    size_t &threadBufferSize = threadPipelineManager->firstBufferSize;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest SaveFileBuffer start buffer: %{public}p,"
                 "pipeline: %{public}p, thread_id:%{public}lu, threadBufferSize:%{public}zu",
                 threadBuffer, threadPipeline, thisId, threadBufferSize);
    MultiPipelineRenderFrame();
    napi_value napiValue = nullptr;
    void *arrayBufferData = nullptr;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest SaveFileBuffer start threadBufferSize:%{public}zu", threadBufferSize);
    napi_status status = napi_create_arraybuffer(env, threadBufferSize, &arrayBufferData, &napiValue);
    if (status != napi_ok || arrayBufferData == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest OH_AudioSuiteEngine_RenderFrame status: %{public}d", static_cast<int>(status));
        if (threadBuffer != nullptr) {
            free(threadBuffer);
            threadBuffer = nullptr;
        }
        // 创建 ArrayBuffer 失败， 返回一个大小为 0 的 ArrayBuffer
        napi_create_arraybuffer(env, 0, &arrayBufferData, &napiValue);
        return napiValue;
    } else {
        if ((threadPipeline == nullptr || threadBuffer == nullptr)) {
            napi_create_arraybuffer(env, 0, &arrayBufferData, &napiValue);
            return napiValue;
        }
        std::copy(static_cast<const char*>(threadBuffer),
            static_cast<const char*>(threadBuffer) + threadBufferSize,
            static_cast<char*>(arrayBufferData));
        if (threadBuffer != nullptr) {
            free(threadBuffer);
            threadBuffer = nullptr;
        }
        return napiValue;
    }
}

Node MultiCreateNodeByType(std::string uuid, OH_AudioNode_Type nodeType)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest CreateNodeByType start, "
                 "threadNodeManager:%{public}p",
                 threadNodeManager.get());
    OH_AudioSuite_Result result = threadNodeManager->createNode(uuid, nodeType);
    if (result != AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest---create AudioSeparation Node Failed");
    }
    Node node = threadNodeManager->GetNodeById(uuid);
    return node;
}

napi_value MultiAddNoiseReduction(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest---addNoiseReduction IN");
    size_t argc = 3;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // 获取uuid
    std::string uuidStr;
    napi_status status = ParseNapiString(env, argv[FIRST_ARGV_PARAM], uuidStr);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest---uuid==%{public}s",
                 uuidStr.c_str());

    // 获取二参inputId
    std::string inputIdStr;
    status = ParseNapiString(env, argv[SECOND_ARGV_PARAM], inputIdStr);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest---inputId==%{public}s",
                 inputIdStr.c_str());

    // 获取当前选中的节点id
    std::string selectNodeId;
    status = ParseNapiString(env, argv[THIRD_ARGV_PARAM], selectNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest---addNoiseReduction---selectNodeId==%{public}s", selectNodeId.c_str());

    napi_value ret = nullptr;
    napi_create_int32(env, 1, &ret);
    Node node = MultiCreateNodeByType(uuidStr, OH_AudioNode_Type::EFFECT_NODE_TYPE_NOISE_REDUCTION);
    if (node.physicalNode == nullptr) {
        return ret;
    }

    int insertRes = -1;
    if (selectNodeId.empty()) {
        insertRes = MultiAddEffectNodeToNodeManager(inputIdStr, uuidStr);
    } else {
        insertRes = threadPipelineManager->nodeManager->insertNode(uuidStr, selectNodeId, Direction::LATER);
    }

    if (insertRes != 0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest---AddEffectNodeToNodeManager ERROR!");
        return ret;
    }
    napi_create_int32(env, 0, &ret);
    return ret;
}

int MultiAddVBEffectNode(std::string inputId, int mode, std::string voiceBeautifierId, std::string selectNodeId)
{
    OH_VoiceBeautifierType type;
    switch (mode) {
        case VB_MODE_CLEAR:
            type = OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CLEAR;
            break;
        case VB_MODE_THEATRE:
            type = OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_THEATRE;
            break;
        case VB_MODE_CD:
            type = OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CD;
            break;
        case VB_MODE_RECORDING_STUDIO:
            type = OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_RECORDING_STUDIO;
            break;
        default:
            type = OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CLEAR;
            break;
    }
    Node node = CreateNodeByType(voiceBeautifierId, OH_AudioNode_Type::EFFECT_NODE_TYPE_VOICE_BEAUTIFIER);
    bool bypass = mode == 0;
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_BypassEffectNode(node.physicalNode, bypass);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
            "audioEditTest---startVBEffect OH_AudioSuiteEngine_BypassEffectNode ERROR %{public}zd", result);
        return result;
    }
    if (bypass) {
        return result;
    }
    result = OH_AudioSuiteEngine_SetVoiceBeautifierType(node.physicalNode, type);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                     "audioEditTest---startVBEffect OH_AudioSuiteEngine_SetVoiceBeautifierType ERROR!");
        return result;
    }
    int res = -1;
    if (selectNodeId.empty()) {
        res = AddEffectNodeToNodeManager(inputId, voiceBeautifierId);
    } else {
        res = g_nodeManager->insertNode(voiceBeautifierId, selectNodeId, Direction::LATER);
    }
    if (res != 0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "AddEffectNodeToNodeManager ERROR!");
        return res;
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "startVBEffect: operation success");
    return result;
}

napi_value MultiStartVBEffect(napi_env env, napi_callback_info info)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "MultiStartVBEffect start, threadNodeManager:%{public}p", threadNodeManager.get());
    size_t argc = 4;
    napi_value argv[4] = {nullptr, nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // inputId
    std::string inputId;
    napi_status status = ParseNapiString(env, argv[ARG_0], inputId);
    // 获取二参、美化类型
    int mode = -1;
    napi_get_value_int32(env, argv[ARG_1], &mode);
    // 获取三参、效果节点id
    std::string voiceBeautifierId;
    status = ParseNapiString(env, argv[ARG_2], voiceBeautifierId);
    // 获取当前选中的节点id
    std::string selectNodeId;
    status = ParseNapiString(env, argv[ARG_3], selectNodeId);
    //调用添加美化效果节点接口
    napi_value ret;
    int result = AddVBEffectNode(inputId, mode, voiceBeautifierId, selectNodeId);
    napi_create_int64(env, result, &ret);
    return ret;
}

napi_value MultiStartFieldEffect(napi_env env, napi_callback_info info)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "MultiStartFieldEffect start, threadNodeManager:%{public}p", threadNodeManager.get());
    size_t argc = 4;
    napi_value argv[4] = {nullptr, nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    std::string inputId;
    std::string fieldEffectId;
    std::string selectedNodeId;
    unsigned int mode = -1;
    napi_status status = ParseNapiString(env, argv[FIRST_ARGV_PARAM], inputId);
    napi_get_value_uint32(env, argv[SECOND_ARGV_PARAM], &mode);
    status = ParseNapiString(env, argv[THIRD_ARGV_PARAM], fieldEffectId);
    status = ParseNapiString(env, argv[FORTH_ARGV_PARAM], selectedNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
        "audioEditTest startFieldEffect inputId:%{public}s"
        ",mode:%{public}u,fieldEffectId:%{public}s,selectedNodeId is %{public}s",
        inputId.c_str(), mode, fieldEffectId.c_str(), selectedNodeId.c_str());
    OH_SoundFieldType type = getSoundFieldTypeByNum(mode);
    napi_value ret;
    Node node = MultiCreateNodeByType(fieldEffectId, OH_AudioNode_Type::EFFECT_NODE_TYPE_SOUND_FIELD);
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_SetSoundFieldType(node.physicalNode, type);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "SetSoundFiledType ERROR!");
        napi_create_int64(env, result, &ret);
        return ret;
    }
    if (selectedNodeId.empty()) {
        int res = MultiAddEffectNodeToNodeManager(inputId, fieldEffectId);
        if (res != 0) {
            napi_create_int64(env, res, &ret);
            return ret;
        }
    } else {
        result = threadNodeManager->insertNode(fieldEffectId, selectedNodeId, Direction::LATER);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            napi_create_int64(env, result, &ret);
            return ret;
        }
    }

    napi_create_int64(env, result, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "startFieldEffect: operation success");
    return ret;
}

napi_value MultiAddAudioSeparation(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest---addAudioSeparation---IN");
    size_t argc = 4;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    // 获取一参
    unsigned int arg1 = 0;
    napi_get_value_uint32(env, argv[FIRST_ARGV_PARAM], &arg1);
    // 获取二参uuid
    std::string uuidStr;
    napi_status status = ParseNapiString(env, argv[SECOND_ARGV_PARAM], uuidStr);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest---uuid==%{public}s",
                 uuidStr.c_str());

    // 获取三参inputId
    std::string inputIdStr;
    status = ParseNapiString(env, argv[THIRD_ARGV_PARAM], inputIdStr);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest---inputId==%{public}s",
                 inputIdStr.c_str());

    // 获取四参
    std::string selectedNodeId;
    status = ParseNapiString(env, argv[FORTH_ARGV_PARAM], selectedNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest addAudioSeparation selectedNodeId is %{public}s", selectedNodeId.c_str());

    napi_value ret;
    napi_create_int64(env, AUDIOSUITE_ERROR_SYSTEM_CODE, &ret);
    Node node = MultiCreateNodeByType(uuidStr, OH_AudioNode_Type::EFFECT_MULTII_OUTPUT_NODE_TYPE_AUDIO_SEPARATION);
    if (node.physicalNode == nullptr) {
        return ret;
    }

    if (selectedNodeId.empty()) {
        int insertRes = MultiAddEffectNodeToNodeManager(inputIdStr, uuidStr);
        if (insertRes == -1) {
            return ret;
        }
    } else {
        OH_AudioSuite_Result result =
            (threadPipelineManager->nodeManager)->insertNode(uuidStr, selectedNodeId, Direction::LATER);
    }
    threadPipelineManager->multiRenderFrameFlag = true;
    napi_create_int64(env, 0, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "audioEditTest---addAudioSeparation: operation success");
    return ret;
}

napi_value MultiStartEnvEffect(napi_env env, napi_callback_info info)
{
    std::shared_ptr<NodeManager> &threadNodeManager = threadPipelineManager->nodeManager;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "MultiStartEnvEffect start, threadNodeManager:%{public}p", threadNodeManager.get());
    size_t argc = 4;
    napi_value *argv = new napi_value[argc];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    napi_status status;
    std::string inputIdStr;
    std::string uuidStr;
    std::string selectedNodeId;
    unsigned int mode = 0;
    status = ParseNapiString(env, argv[FIRST_ARGV_PARAM], inputIdStr);
    status = ParseNapiString(env, argv[SECOND_ARGV_PARAM], uuidStr);
    napi_get_value_uint32(env, argv[THIRD_ARGV_PARAM], &mode);
    status = ParseNapiString(env, argv[FORTH_ARGV_PARAM], selectedNodeId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "inputId:%{public}s,uuid:%{public}s,mode:"
        "%{public}u,selectedNodeId:%{public}s", inputIdStr.c_str(), uuidStr.c_str(), mode, selectedNodeId.c_str());
    OH_EnvironmentType type = GetEnvEnumByNumber(mode);
    napi_value ret;
    Node node = MultiCreateNodeByType(uuidStr, OH_AudioNode_Type::EFFECT_NODE_TYPE_ENVIRONMENT_EFFECT);
    if (node.physicalNode == nullptr) {
        napi_create_int64(env, AUDIOSUITE_ERROR_SYSTEM_CODE, &ret);
        return ret;
    }
    OH_AudioSuite_Result result = OH_AudioSuiteEngine_SetEnvironmentType(node.physicalNode, type);
    if (result != AUDIOSUITE_SUCCESS) {
        napi_create_int64(env, result, &ret);
        return ret;
    }
    if (selectedNodeId.empty()) {
        int insertRes = MultiAddEffectNodeToNodeManager(inputIdStr, uuidStr);
        if (insertRes == -1) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "AddEffectNodeToNodeManager ERROR!");
            napi_create_int64(env, insertRes, &ret);
            return ret;
        }
    } else {
        result = threadNodeManager->insertNode(uuidStr, selectedNodeId, Direction::LATER);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "startEnvEffect insertNode ERROR!");
            napi_create_int64(env, result, &ret);
            return ret;
        }
    }
    napi_create_int64(env, 0, &ret);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "MultiStartEnvEffect: operation success");
    return ret;
}

napi_value MultiGetSecondOutputAudio(napi_env env, napi_callback_info info)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG, "audioEditTest---getAudioOfTap---IN");
    napi_value napiValue = nullptr;
    void *data;
    napi_create_arraybuffer(env, threadPipelineManager->secondBufferSize, &data, &napiValue);
    char *&secData = threadPipelineManager->secondAudioBuffer;
    size_t &threadSecDataSize = threadPipelineManager->secondBufferSize;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, MULTI_PIPELINE_TAG,
                 "secData:%{public}p, threadSecDataSize:%{public}zu, pipeline:%{public}p",
                 secData, threadSecDataSize, threadPipelineManager->audioSuitePipeline);
    std::copy(static_cast<const char*>(secData),
        static_cast<const char*>(secData) + threadSecDataSize, static_cast<char*>(data));
    std::fill(static_cast<char*>(secData), static_cast<char*>(secData) + threadSecDataSize, 0);
    threadSecDataSize = 0;
    return napiValue;
}