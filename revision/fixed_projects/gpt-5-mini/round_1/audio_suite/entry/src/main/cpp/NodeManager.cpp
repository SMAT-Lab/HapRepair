/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "NodeManager.h"
#include <hilog/log.h>
#include <stack>
#include <sstream>

static const int GLOBAL_RESMGR = 0xFF00;
static const char *TAG = "[AudioEditTestApp_AudioEdit_cpp]";

Node g_defaultNode;

NodeManager::NodeManager(OH_AudioSuitePipeline *audioSuitePipeLine) : audioSuitePipeLine(audioSuitePipeLine)
{
    // 初始化代码
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTest NodeManager audioSuitePipeLine: %{public}p",
        static_cast<void *>(audioSuitePipeLine));
}

NodeManager::~NodeManager()
{}

OH_AudioSuite_Result NodeManager::createNode(
    const std::string &nodeId, const OH_AudioNode_Type nodeType, OH_AudioNodeBuilder *builder)
{
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTest createNode start,OH_AudioNode_Type: %{public}d",
        static_cast<int>(nodeType));
    OH_AudioNode *physicalNode;
    OH_AudioSuite_Result result;

    if (builder == nullptr) {
        result = OH_AudioSuiteNodeBuilder_Create(&builder);
        OH_LOG_Print(LOG_APP,
            LOG_INFO,
            GLOBAL_RESMGR,
            TAG,
            "NodeManagerTest createNode OH_AudioSuiteNodeBuilder_Create result: %{public}d",
            static_cast<int>(result));
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
        result = OH_AudioSuiteNodeBuilder_SetNodeType(builder, nodeType);
        OH_LOG_Print(LOG_APP,
            LOG_INFO,
            GLOBAL_RESMGR,
            TAG,
            "NodeManagerTest createNode OH_AudioSuiteNodeBuilder_SetNodeType result: %{public}d",
            static_cast<int>(result));
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
    }
    result = OH_AudioSuiteEngine_CreateNode(audioSuitePipeLine, builder, &physicalNode);
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTest createNode OH_AudioSuiteEngine_CreateNode result:%{public}d",
        static_cast<int>(result));
    // 构造节点结构体并存储
    if (result == OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        Node node;
        node.id = nodeId;
        node.type = nodeType;
        node.physicalNode = physicalNode;
        nodes[node.id] = node;
    }
    OH_AudioSuiteNodeBuilder_Destroy(builder);
    return result;
}

OH_AudioSuite_Result NodeManager::removeNode(const std::string &nodeId)
{
    if (nodes.find(nodeId) == nodes.end()) {
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    Node node = nodes[nodeId];
    OH_AudioSuite_Result result;

    // Disconnect the old connection first, then connect the new node.
    if (node.preNodeIds.empty() && node.nextNodeId.empty()) {
        result = OH_AudioSuiteEngine_DestroyNode(node.physicalNode);
    } else if (node.preNodeIds.empty()) {
        result = disconnect(node.id, node.nextNodeId);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
        result = OH_AudioSuiteEngine_DestroyNode(node.physicalNode);
    } else if (node.nextNodeId.empty()) {
        result = disconnect(node.preNodeIds[0], node.id);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
        result = OH_AudioSuiteEngine_DestroyNode(node.physicalNode);
    } else {
        result = disconnect(node.preNodeIds[0], node.id);
        result = disconnect(node.id, node.nextNodeId);
        result = connect(node.preNodeIds[0], node.nextNodeId);
        if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
            return result;
        }
        result = OH_AudioSuiteEngine_DestroyNode(node.physicalNode);
    }
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    nodes.erase(node.id);
    OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "removeNode result:%{public}d", static_cast<int>(result));
    return result;
}

OH_AudioSuite_Result NodeManager::moveNode(
    const std::string &sourceNodeId, const std::string &targetNodeId, Direction direction)
{
    if (nodes.find(sourceNodeId) == nodes.end() || nodes.find(targetNodeId) == nodes.end()) {
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    OH_AudioSuite_Result result;
    Node sourceNode = nodes[sourceNodeId];
    Node targetNode = nodes[targetNodeId];
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTset moveNode targetNode:%{pubilc}d",
        static_cast<int>(targetNode.type));
    if (sourceNode.preNodeIds.empty()) {
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    result = disconnect(sourceNode.preNodeIds[0], sourceNode.id);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    result = disconnect(sourceNode.id, sourceNode.nextNodeId);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    result = connect(sourceNode.preNodeIds[0], sourceNode.nextNodeId);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    result = insertNode(sourceNodeId, targetNodeId, direction);
    return result;
}

OH_AudioSuite_Result NodeManager::connect(const std::string &fromId, const std::string &toId)
{
    // 检查是否有该节点
    if (nodes.find(fromId) == nodes.end() || nodes.find(toId) == nodes.end()) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "NodeManagerTest connect:Node ID not found");
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTest connect fromId toId: %{public}s %{public}s",
        fromId.c_str(),
        toId.c_str());
    OH_AudioSuite_Result result;
    Node preNode = nodes[fromId];
    Node nextNode = nodes[toId];

    // 连接之前需要停止管线
    result = stopPipelineState();
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }

    result = OH_AudioSuiteEngine_ConnectNodes(preNode.physicalNode, nextNode.physicalNode);
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTest connect OH_AudioSuiteEngine_ConnectNodes result: %{public}d",
        static_cast<int>(result));
    if (result == OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        nodes[toId].preNodeIds.push_back(fromId);
        nodes[fromId].nextNodeId = toId;
    }
    return result;
}

OH_AudioSuite_Result NodeManager::resetNode(const std::string &nodeId, const OH_AudioNode_Type type)
{
    if (nodes.find(nodeId) == nodes.end()) {
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    OH_AudioSuite_Result result;
    if (result == OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        nodes[nodeId].type = type;
    }
    return result;
}

OH_AudioSuite_Result NodeManager::disconnect(const std::string &fromId, const std::string &toId)
{
    if (nodes.find(fromId) == nodes.end() || nodes.find(toId) == nodes.end()) {
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    OH_AudioSuite_Result result;
    Node preNode = nodes[fromId];
    Node nextNode = nodes[toId];

    // 解连接之前需要停止管线
    result = stopPipelineState();
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }

    result = OH_AudioSuiteEngine_DisconnectNodes(preNode.physicalNode, nextNode.physicalNode);
    OH_LOG_Print(
        LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "NodeManagerTest disconnect: %{public}d", static_cast<int>(result));
    if (result == OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        nodes[fromId].nextNodeId = "";
        auto it = std::find(nodes[toId].preNodeIds.begin(), nodes[toId].preNodeIds.end(), fromId);
        if (it != nodes[toId].preNodeIds.end()) {
            nodes[toId].preNodeIds.erase(it);
        }
    }
    return result;
}

void NodeManager::DisconnectAll(const std::string &nodeId)
{
    // todo:混音节点需要解开所有连接
    nodes[nodeId].nextNodeId = "";
    nodes[nodeId].preNodeIds.clear();
}

const std::unordered_map<std::string, Node> &NodeManager::getAllNodes() const
{
    return nodes;
}

const Node &NodeManager::GetNodeById(const std::string &nodeId) const
{
    auto it = nodes.find(nodeId);
    if (it != nodes.end()) {
        return it->second;
    } else {
        return g_defaultNode;
    }
}

//获取均衡器类型
std::string GetEqualizerOptions(const Node &node)
{
    OH_EqualizerFrequencyBandGains type;
    OH_AudioSuite_Result ret = OH_AudioSuiteEngine_GetEqualizerFrequencyBandGains(node.physicalNode, &type);
    if (ret != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return "0";
    }
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < 10; ++i) { // 均衡器有10个频段
        if (i > 0)
            oss << ",";
        oss << type.gains[i];
    }
    oss << "]";
    return oss.str();
}

//获取声音美化类型
std::string GetVoiceBeautifierOptions(const Node &node)
{
    OH_VoiceBeautifierType type;
    OH_AudioSuite_Result ret = OH_AudioSuiteEngine_GetVoiceBeautifierType(node.physicalNode, &type);
    if (ret != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return "0";
    }
    switch (type) {
        case OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CLEAR:
            return "1";
        case OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_THEATRE:
            return "2";
        case OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_CD:
            return "3";
        case OH_VoiceBeautifierType::VOICE_BEAUTIFIER_TYPE_RECORDING_STUDIO:
            return "4";
        default:
            return "0";
    }
}

//获取声场类型
std::string GetSoundFieldOptions(const Node &node)
{
    OH_SoundFieldType type;
    OH_AudioSuite_Result ret = OH_AudioSuiteEngine_GetSoundFieldType(node.physicalNode, &type);
    if (ret != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return "0";
    }
    switch (type) {
        case OH_SoundFieldType::SOUND_FIELD_FRONT_FACING:
            return "1";
        case OH_SoundFieldType::SOUND_FIELD_GRAND:
            return "2";
        case OH_SoundFieldType::SOUND_FIELD_NEAR:
            return "3";
        case OH_SoundFieldType::SOUND_FIELD_WIDE:
            return "4";
        default:
            return "0";
    }
}

//获取环境类型
std::string GetEnvironmentEffectOptions(const Node &node)
{
    OH_EnvironmentType type;
    OH_AudioSuite_Result ret = OH_AudioSuiteEngine_GetEnvironmentType(node.physicalNode, &type);
    if (ret != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return "0";
    }
    switch (type) {
        case OH_EnvironmentType::ENVIRONMENT_TYPE_BROADCAST:
            return "1";
        case OH_EnvironmentType::ENVIRONMENT_TYPE_EARPIECE:
            return "2";
        case OH_EnvironmentType::ENVIRONMENT_TYPE_UNDERWATER:
            return "3";
        case OH_EnvironmentType::ENVIRONMENT_TYPE_GRAMOPHONE:
            return "4";
        default:
            return "0";
    }
}

std::string NodeManager::GetOptionsByType(const Node &node)
{
    switch (node.type) {
        case OH_AudioNode_Type::EFFECT_NODE_TYPE_EQUALIZER:
            return GetEqualizerOptions(node);
        case OH_AudioNode_Type::EFFECT_NODE_TYPE_VOICE_BEAUTIFIER:
            return GetVoiceBeautifierOptions(node);
        case OH_AudioNode_Type::EFFECT_NODE_TYPE_SOUND_FIELD:
            return GetSoundFieldOptions(node);
        case OH_AudioNode_Type::EFFECT_NODE_TYPE_ENVIRONMENT_EFFECT:
            return GetEnvironmentEffectOptions(node);
        default:
            return "";
    }
}

const std::vector<Node> NodeManager::getNodesByType(const OH_AudioNode_Type targetType) const
{
    std::vector<Node> result = {};
    for (const auto &pair : nodes) {
        OH_LOG_Print(LOG_APP,
            LOG_INFO,
            GLOBAL_RESMGR,
            TAG,
            "NodeManagerTest getNodesByType node of nodes: %{public}d",
            static_cast<int>(pair.second.type));
        if (pair.second.type == targetType) {
            result.push_back(pair.second);
        }
    }
    return result;
}

bool NodeManager::IsValidNode(const std::string &nodeId)
{
    return nodes.find(nodeId) != nodes.end();
}

OH_AudioSuite_Result NodeManager::insertBefore(
    const std::string &sourceNodeId, const std::string &targetNodeId, const Node &targetNode)
{
    if (targetNode.preNodeIds.empty()) {
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }

    OH_AudioSuite_Result result = disconnect(targetNode.preNodeIds[0], targetNodeId);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }

    result = connect(targetNode.preNodeIds[0], sourceNodeId);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }

    return connect(sourceNodeId, targetNodeId);
}

OH_AudioSuite_Result NodeManager::insertAfter(
    const std::string &sourceNodeId, const std::string &targetNodeId, const Node &targetNode)
{
    OH_AudioSuite_Result result = disconnect(targetNodeId, targetNode.nextNodeId);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }

    result = connect(targetNodeId, sourceNodeId);
    if (result != OH_AudioSuite_Result::AUDIOSUITE_SUCCESS) {
        return result;
    }
    return connect(sourceNodeId, targetNode.nextNodeId);
}

OH_AudioSuite_Result NodeManager::insertNode(
    const std::string &sourceNodeId, const std::string &targetNodeId, Direction direction)
{
    // direction为BEFORE，即插入到targetNodeId之前
    if (!IsValidNode(sourceNodeId) || !IsValidNode(targetNodeId)) {
        return OH_AudioSuite_Result::AUDIOSUITE_ERROR_INVALID_PARAM;
    }
    Node targetNode = nodes[targetNodeId];
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTest insertNode targetNode: %{public}d",
        static_cast<int>(targetNode.type));
    if (direction == Direction::BEFORE) {
        return insertBefore(sourceNodeId, targetNodeId, targetNode);
    } else {
        return insertAfter(sourceNodeId, targetNodeId, targetNode);
    }
}

// 节点编排前（连接或者解连接）需要pipeline状态为stopped
OH_AudioSuite_Result NodeManager::stopPipelineState()
{
    OH_AudioSuite_Result result;
    OH_AudioSuite_PipelineState pipelineState;
    result = OH_AudioSuiteEngine_GetPipelineState(audioSuitePipeLine, &pipelineState);
    if (pipelineState == OH_AudioSuite_PipelineState::AUDIOSUITE_PIPELINE_RUNNING) {
        result = OH_AudioSuiteEngine_StopPipeline(audioSuitePipeLine);
    }
    return result;
}

char *NodeManager::getNodeTypeName(OH_AudioNode_Type type)
{
    switch (type) {
        case OH_AudioNode_Type::INPUT_NODE_TYPE_DEFAULT:
            return "inputNode";
        case OH_AudioNode_Type::EFFECT_NODE_TYPE_AUDIO_MIXER:
            return "mixerNode";
        default:
            return "effectNode";
    }
}

void NodeManager::GetPipeLineDetail()
{
    std::vector<Node> outputNodes = getNodesByType(OH_AudioNode_Type::OUTPUT_NODE_TYPE_DEFAULT);
    if (outputNodes.size() != 1) {
        OH_LOG_Print(LOG_APP,
            LOG_INFO,
            GLOBAL_RESMGR,
            TAG,
            "NodeManagerTest GetPipeLineDetail outputNodes size: %{public}d",
            static_cast<int>(outputNodes.size()));
        return;
    }
    std::string pipelineDetails = "outputNode ";
    std::stack<Node> nodeStack;
    nodeStack.push(outputNodes[0]);

    while (!nodeStack.empty()) {
        Node currentNode = nodeStack.top();
        nodeStack.pop();

        if (currentNode.preNodeIds.empty()) {
            continue;
        }

        pipelineDetails += "[";
        for (auto it = currentNode.preNodeIds.rbegin(); it != currentNode.preNodeIds.rend(); ++it) {
            Node childNode = GetNodeById(*it);
            char *nodeType = getNodeTypeName(childNode.type);
            pipelineDetails += nodeType;
            nodeStack.push(childNode);
        }
        pipelineDetails += "] ";
    }
    OH_LOG_Print(LOG_APP,
        LOG_INFO,
        GLOBAL_RESMGR,
        TAG,
        "NodeManagerTest GetPipeLineDetail: %{public}s",
        pipelineDetails.c_str());
}