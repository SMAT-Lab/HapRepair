/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include <thread>
#include "Input.h"
#include "./utils/Utils.h"
#include "./Output.h"
#include "hilog/log.h"

#include <multimedia/player_framework/native_avcodec_base.h>

const int GLOBAL_RESMGR = 0xFF00;
const char *INPUT_TAG = "[AudioEditTestApp_Input_cpp]";

OH_AudioSuitePipeline *g_audioSuitePipeline = nullptr;

OH_AudioSuiteEngine *g_audioSuiteEngine = nullptr;

char *g_totalBuff = (char *)malloc(8 * 1024 * 1024);

int32_t g_totalSize = 0;

std::map<std::string, std::vector<uint8_t>> g_writeDataBufferMap = {};

std::map<std::string, UserData *> g_userDataMap = {};

OH_AudioFormat g_audioFormatInput = {
    .encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW
};

// 创造 output builder 构造器
OH_AudioNodeBuilder *builderOut = nullptr;

napi_status ParseArguments(napi_env env, napi_value *argv, AudioParams &params)
{
    napi_status status = ParseNapiString(env, argv[ARG_0], params.inputId);
    status = ParseNapiString(env, argv[ARG_1], params.outputId);
    status = ParseNapiString(env, argv[ARG_2], params.mixerId);
    OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, INPUT_TAG,
        "inputId: %{public}s, outputId: %{public}s, mixerId: %{public}s",
        params.inputId.c_str(), params.outputId.c_str(), params.mixerId.c_str());
    status = napi_get_value_uint32(env, argv[ARG_3], &params.fd);
    status = napi_get_value_uint32(env, argv[ARG_4], &params.fileLength);
    OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, INPUT_TAG,
        "fd: %{public}d, fileLength: %{public}d, status: %{public}d",
        params.fd, params.fileLength, status);
    return status;
}

void ResetAllIsResetTotalWriteAudioDataSize()
{
    for (auto &pair : g_userDataMap) {
        pair.second->isResetTotalWriteAudioDataSize = true;
    }
}

bool GetAudioProperties(OH_AVFormat *trackFormat, int32_t &sampleRate, int32_t &channels, int32_t &bitsPerSample)
{
    if (!OH_AVFormat_GetIntValue(trackFormat, OH_MD_KEY_AUD_SAMPLE_RATE, &sampleRate)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG, "get sample rate failed");
        return false;
    }
    if (!OH_AVFormat_GetIntValue(trackFormat, OH_MD_KEY_AUD_CHANNEL_COUNT, &channels)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG, "get channel count failed");
        return false;
    }
    if (!OH_AVFormat_GetIntValue(trackFormat, OH_MD_KEY_AUDIO_SAMPLE_FORMAT, &bitsPerSample)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG, "get bits per sample failed");
        return false;
    }
    OH_LOG_Print(LOG_APP, LOG_WARN, GLOBAL_RESMGR, INPUT_TAG,
        "sampleRate: %{public}d, channels: %{public}d, bitsPerSample: %{public}d", sampleRate, channels, bitsPerSample);
    // 设置采样率
    g_audioFormatInput.samplingRate = SetSamplingRate(sampleRate);
    // 设置声道
    g_audioFormatInput.channelCount = channels;
    g_audioFormatInput.channelLayout = SetChannelLayout(channels);
    // 设置位深
    g_audioFormatInput.sampleFormat = SetSampleFormat(bitsPerSample);
    // 设置编码格式
    g_audioFormatInput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;
    
    g_audioFormatOutput.samplingRate = g_audioFormatInput.samplingRate;
    g_audioFormatOutput.channelCount = g_audioFormatInput.channelCount;
    g_audioFormatOutput.channelLayout = g_audioFormatInput.channelLayout;
    g_audioFormatOutput.sampleFormat = g_audioFormatInput.sampleFormat;
    g_audioFormatOutput.encodingType = g_audioFormatInput.encodingType;

    return true;
}

void ReadTrackSamples(OH_AVDemuxer *demuxer, uint32_t trackIndex, int bufferSize,
    std::atomic<bool>& isEnd, std::atomic<bool>& threadFinished)
{
    g_totalSize = 0;
    g_totalBuff = nullptr;
    // 添加解封装轨道
    if (OH_AVDemuxer_SelectTrackByID(demuxer, trackIndex) != AV_ERR_OK) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG, "select audio track failed: %{public}d", trackIndex);
    }
    // 创建缓冲区
    if (bufferSize <= 0) {
        return;
    }
    OH_AVBuffer *pcmBuffer = OH_AVBuffer_Create(bufferSize);
    char *totalBuffer = (char *)malloc(bufferSize);
    if (pcmBuffer == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG, "create pcmBuffer failed");
    }
    OH_AVCodecBufferAttr info;
    int32_t ret;

    while (!isEnd.load()) {
        ret = OH_AVDemuxer_ReadSampleBuffer(demuxer, trackIndex, pcmBuffer);
        if (ret == AV_ERR_OK) {
            OH_AVBuffer_GetBufferAttr(pcmBuffer, &info);
            // 将当前样本的数据复制到 totalBuff 中
            std::copy(reinterpret_cast<char *>(OH_AVBuffer_GetAddr(pcmBuffer)),
                reinterpret_cast<char *>(OH_AVBuffer_GetAddr(pcmBuffer)) + info.size, totalBuffer + g_totalSize);
            g_totalSize += info.size;
            if (info.flags == OH_AVCodecBufferFlags::AVCODEC_BUFFER_FLAGS_EOS) {
                isEnd.store(true);
            }
        } else {
            OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
                "get pcmBuffer failed, ret: %{public}d, trackIndex: %{public}d", ret, trackIndex);
        }
    }
    g_totalBuff = (char *)malloc(g_totalSize);
    std::copy(totalBuffer, totalBuffer + g_totalSize, g_totalBuff);
    // 销毁缓冲区
    free(totalBuffer);
    OH_AVBuffer_Destroy(pcmBuffer);
    threadFinished.store(true);
}

void RunAudioThread(OH_AVDemuxer *demuxer, int32_t fileLength)
{
    std::atomic<bool> audioIsEnd{false};
    std::atomic<bool> audioThreadFinished{false};

    std::thread audioThread(ReadTrackSamples, demuxer, 0, fileLength,
        std::ref(audioIsEnd), std::ref(audioThreadFinished));
    audioThread.join();
}

void StoreTotalBuffToMap(const char *totalBuff, int32_t size, const std::string &key)
{
    if (size > 0 && totalBuff != nullptr) {
        std::vector<uint8_t> buffer(totalBuff, totalBuff + size);
        g_writeDataBufferMap[key] = buffer;
        return;
    }
    OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG, "audioEditTest StoreTotalBuffToMap failed");
}

void CreateInputNode(napi_env env, const std::string &inputId, napi_value &napiValue, OH_AudioSuite_Result &result)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG, "audioEditTest CreateInputNode start");
    // 添加音频，将音频的buffer出存储到map中
    StoreTotalBuffToMap(g_totalBuff, g_totalSize, inputId);
    auto it = g_writeDataBufferMap.find(inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest AudioInAndOutInit g_writeDataBufferMap[inputId] length: %{public}d", it->second.size());
    // 创造 builder 构造器
    OH_AudioNodeBuilder *builderIn;
    result = OH_AudioSuiteNodeBuilder_Create(&builderIn);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest OH_AudioSuiteNodeBuilder_Create result: %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        return;
    }
    // 透传节点类型
    result = OH_AudioSuiteNodeBuilder_SetNodeType(builderIn, OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "NodeManagerTest createNode OH_AudioSuiteNodeBuilder_SetNodeType result: %{public}d",
        static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        return;
    }

    // 封装方法，设置 音频文件的 参数 以及 写入音频文件到缓冲区
    result = SetParamsAndWriteData(builderIn, inputId, OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest SetParamsAndWriteData result: %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        napi_create_int64(env, static_cast<int>(result), &napiValue);
        return;
    }

    // 创建input节点
    g_nodeManager->createNode(inputId, OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT, builderIn);
}

OH_AudioSuite_Result SetParamsAndWriteData(OH_AudioNodeBuilder *builder, std::string inputId, OH_AudioNode_Type type)
{
    OH_AudioSuite_Result result = OH_AudioSuiteNodeBuilder_SetFormat(builder, g_audioFormatInput);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest OH_AudioNodeBuilder_SetFormat result is %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    if (type != OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT) {
        return result;
    }
    UserData *data = new UserData();
    data->id = inputId;
    // 后面可以考虑去掉g_totalSize, 用入参形式传入
    data->bufferSize = g_totalSize;
    data->totalWriteAudioDataSize = 0;
    data->isResetTotalWriteAudioDataSize = false;
    void *userData = data;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest OH_AudioSuiteNodeBuilder_SetRequestDataCallback data address is %{public}p", &data);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest OH_AudioNodeBuilder_SetFormat userData inputId is %{public}s",
        static_cast<UserData *>(userData)->id.c_str());
    // 设置OH_AudioSuiteNodeBuilder_SetRequestDataCallback回调，创建节点之前
    result = OH_AudioSuiteNodeBuilder_SetRequestDataCallback(builder, WriteDataCallBack, userData);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest OH_AudioSuiteNodeBuilder_SetRequestDataCallback result is %{public}d",
        static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    // 将UserData实例存入映射表中
    g_userDataMap[inputId] = data;
    return result;
}

bool CheckParameters(OH_AudioNode *audioNode, void *audioData, bool *finished)
{
    if (audioNode == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest WriteDataCallBack audioNode is nullptr");
        *finished = true;
        return false;
    }
    if (audioData == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest WriteDataCallBack audioData is nullptr");
        *finished = true;
        return false;
    }
    if (finished == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest WriteDataCallBack finished is nullptr");
        *finished = true;
        return false;
    }
    return true;
}

int32_t WriteDataCallBack(OH_AudioNode *audioNode, void *userData, void *audioData,
    int32_t audioDataSize, bool *finished)
{
    if (!CheckParameters(audioNode, audioData, finished)) {
        return 0;
    }
    // 处理音频数据 此处如果是nullptr，是demo获取音频数据的问题，非底层接口问题
    std::string inputId = static_cast<UserData *>(userData)->id;
    auto usetDataIt = g_userDataMap.find(inputId);
    if (usetDataIt->second->isResetTotalWriteAudioDataSize) {
        usetDataIt->second->isResetTotalWriteAudioDataSize = false;
        static_cast<UserData *>(userData)->totalWriteAudioDataSize = 0;
    }
    int32_t totalSize = usetDataIt->second->bufferSize;
    ssize_t totalWriteAudioDataSize = usetDataIt->second->totalWriteAudioDataSize;
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG, "audioEditTest WriteDataCallBack inputId: %{public}s",
        inputId.c_str());
    auto it = g_writeDataBufferMap.find(inputId);
    if (it == g_writeDataBufferMap.end()) {
        // map没有找到对应的音频buffer
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest WriteDataCallBack g_writeDataBufferMap is end");
        *finished = true;
        return 0;
    }
    // 计算剩余数据量
    int32_t remainingDataSize = totalSize - totalWriteAudioDataSize;
    // 确定本次写入的实际数据量
    int32_t actualDataSize = std::min(audioDataSize, remainingDataSize);
    // 将数据从totalBuff_复制到audioData
    std::copy(it->second.data() + totalWriteAudioDataSize,
        it->second.data() + totalWriteAudioDataSize + actualDataSize, static_cast<char *>(audioData));
    // 跟新已写入的数据量
    totalWriteAudioDataSize += actualDataSize;
    usetDataIt->second->totalWriteAudioDataSize = totalWriteAudioDataSize;
    // 如果不够，则补0
    int32_t padSize = audioDataSize - remainingDataSize;
    if (padSize > 0) {
        std::fill_n(static_cast<char *>(audioData) + actualDataSize, padSize, 0);
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest WriteDataCallBack totalSize: %{public}d, totalWriteAudioDataSize: %{public}d, "
        "audioDataSize: %{public}d, actualDataSize:%{public}d, padSize: %{public}d",
        totalSize, totalWriteAudioDataSize, audioDataSize, actualDataSize, padSize);
    // 如果所有数据都写入完毕
    if (totalWriteAudioDataSize >= totalSize) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG, "audioEditTest WriteDataCallBack is finished");
        g_totalSize = 0;
        totalWriteAudioDataSize = 0;
        *finished = true;
    }
    // 返回写入的数据数据量
    return actualDataSize;
}

void UpdateInputNode(napi_value &napiValue, OH_AudioSuite_Result &result, const UpdateInputNodeParams &params)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG, "audioEditTest UpdateInputNode start");
    // 设置采样率
    g_audioFormatInput.samplingRate = SetSamplingRate(params.sampleRate);
    // 设置声道
    g_audioFormatInput.channelCount = params.channels;
    g_audioFormatInput.channelLayout = SetChannelLayout(params.channels);
    // 设置位深
    g_audioFormatInput.sampleFormat = SetSampleFormat(params.bitsPerSample);
    // 设置编码格式
    g_audioFormatInput.encodingType = OH_Audio_EncodingType::AUDIO_ENCODING_TYPE_RAW;

    g_audioFormatOutput.samplingRate = g_audioFormatInput.samplingRate;
    g_audioFormatOutput.channelCount = g_audioFormatInput.channelCount;
    g_audioFormatOutput.channelLayout = g_audioFormatInput.channelLayout;
    g_audioFormatOutput.sampleFormat = g_audioFormatInput.sampleFormat;
    g_audioFormatOutput.encodingType = g_audioFormatInput.encodingType;
    
    const std::vector<Node> inPutNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT);
    result = OH_AudioSuiteEngine_SetAudioFormat(inPutNodes[0].physicalNode, &g_audioFormatInput);
    const std::vector<Node> outPutNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    result = OH_AudioSuiteEngine_SetAudioFormat(outPutNodes[0].physicalNode, &g_audioFormatOutput);
    // 添加音频，将音频的buffer出存储到map中，，上一行中的memcpy可以考虑删除了
    if (g_writeDataBufferMap.find(params.inputId) != g_writeDataBufferMap.end()) {
        // 键存在，执行删除操作
        g_writeDataBufferMap.erase(params.inputId);
    }
    StoreTotalBuffToMap(g_totalBuff, g_totalSize, params.inputId);
    auto it = g_writeDataBufferMap.find(params.inputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest AudioInAndOutInit g_writeDataBufferMap[inputId] length: %{public}d", it->second.size());
    UserData *data = new UserData();
    data->id = params.inputId;
    // 后面可以考虑去掉g_totalSize，用入参形式传入
    data->bufferSize = g_totalSize;
    data->totalWriteAudioDataSize = 0;
    data->isResetTotalWriteAudioDataSize = false;
    // 将UserData实例存入映射表中
    if (g_userDataMap.find(params.inputId) != g_userDataMap.end()) {
        // 键存在，执行删除操作
        g_userDataMap.erase(params.inputId);
    }
    g_userDataMap[params.inputId] = data;
}

void ManageOutputNodes(napi_env env, const std::string &inputId,
    const std::string &outputId, const std::string &mixerId, OH_AudioSuite_Result &result)
{
    const std::vector<Node> outPutNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    if (outPutNodes.size() > 0) {
        ManageExistingOutputNodes(inputId, mixerId, result, outPutNodes);
    } else {
        CreateAndConnectOutputNodes(inputId, outputId, result);
    }
}

void ManageExistingOutputNodes(const std::string &inputId, const std::string &mixerId,
    OH_AudioSuite_Result &result, std::vector<Node> outPutNodes)
{
    const std::vector<Node> mixerNodes = g_nodeManager->getNodesByType(OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER);
    if (mixerNodes.size() > 0) {
        result = g_nodeManager->connect(inputId, mixerNodes[0].id);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest connect input and mixer result: %{public}d", static_cast<int>(result));
    } else {
        result = g_nodeManager->createNode(mixerId, OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest nodeManagerCreateMixerNode result: %{public}d", static_cast<int>(result));

        result = g_nodeManager->insertNode(mixerId, outPutNodes[0].id, Direction::BEFORE);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest insertMixerNode result: %{public}d", static_cast<int>(result));

        result = g_nodeManager->connect(inputId, mixerId);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
            "audioEditTest connect inputId and mixerId result: %{public}d", static_cast<int>(result));
    }
}

void CreateAndConnectOutputNodes(const std::string &inputId, const std::string &outputId, OH_AudioSuite_Result &result)
{
    result = OH_AudioSuiteNodeBuilder_Create(&builderOut);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest OH_AudioSuiteNodeBuilder_Create output builder result: %{public}d",
        static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return;
    }

    result = OH_AudioSuiteNodeBuilder_SetNodeType(builderOut, OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "NodeManagerTest createNode OH_AudioSuiteNodeBuilder_SetNodeType result: %{public}d",
        static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return;
    }
    // 封装方法，设置 音频文件的 参数 以及 写入音频文件到缓冲区
    result = SetParamsAndWriteData(builderOut, inputId, OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest SetParamsAndWriteData result: %{public}d", static_cast<int>(result));
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return;
    }

    result = g_nodeManager->createNode(outputId, OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT, builderOut);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest nodeManagerCreateOutputNode result: %{public}d", static_cast<int>(result));

    result = g_nodeManager->connect(inputId, outputId);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, INPUT_TAG,
        "audioEditTest nodeManagerConnectInputAndOutput result: %{public}d", static_cast<int>(result));
}