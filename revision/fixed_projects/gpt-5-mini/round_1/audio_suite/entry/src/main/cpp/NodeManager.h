/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef AUDIOEDITTESTAPP_NODEMANAGER_H
#define AUDIOEDITTESTAPP_NODEMANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include "ohaudio/native_audio_suite_base.h"
#include "ohaudio/native_audio_suite_engine.h"

enum NodeType { INPUT_NODE, OUTPUT_NODE, NOISE_REDUCTION_NODE, AUDIO_SEPARATION_NODE, SOUND_FIELD_NODE, MIXER_NODE };

enum Direction { BEFORE, LATER };

struct Node {
    std::string id;
    OH_AudioNode_Type type;
    std::vector<std::string> preNodeIds = {};  // 前序节点ID列表
    std::string nextNodeId = "";               // 后续节点ID列表
    OH_AudioNode *physicalNode = nullptr;      // 真实节点
};

class NodeManager {
private:
    std::unordered_map<std::string, Node> nodes;  // 节点存储，键为节点ID
    OH_AudioSuitePipeline *audioSuitePipeLine;    // 一个NodeManager一个管线

    // 获取管线连接详情
    void GetPipeLineDetail();

public:
    explicit NodeManager(OH_AudioSuitePipeline *audioSuitePipeLine);
    ~NodeManager();

    // 创建节点：创建真实节点，并构造Node，然后存入nodes
    OH_AudioSuite_Result createNode(
        const std::string &nodeId, const OH_AudioNode_Type NodeType, OH_AudioNodeBuilder *builder = nullptr);

    // 删除节点
    OH_AudioSuite_Result removeNode(const std::string &nodeId);

    // 重新设置效果节点
    OH_AudioSuite_Result resetNode(const std::string &nodeId, const OH_AudioNode_Type type);

    // 连接两个节点
    OH_AudioSuite_Result connect(const std::string &fromId, const std::string &toId);

    // 断开两个节点的连接
    OH_AudioSuite_Result disconnect(const std::string &fromId, const std::string &toId);

    // 断开与指定节点的所有连接
    void DisconnectAll(const std::string &nodeId);

    // 插入节点
    OH_AudioSuite_Result insertNode(
        const std::string &sourceNodeId, const std::string &targetNodeId, Direction direction);

    // 移动节点
    OH_AudioSuite_Result moveNode(
        const std::string &sourceNodeId, const std::string &targetNodeId, Direction direction);

    // 获取所有节点
    const std::unordered_map<std::string, Node> &getAllNodes() const;

    // 根据id获取节点
    const Node &GetNodeById(const std::string &nodeId) const;

    // 根据节点type获取节点
    const std::vector<Node> getNodesByType(const OH_AudioNode_Type type) const;

    // 停止管线状态
    OH_AudioSuite_Result stopPipelineState();

    bool IsValidNode(const std::string &nodeId);

    OH_AudioSuite_Result insertBefore(
        const std::string &sourceNodeId, const std::string &targetNodeId, const Node &targetNode);

    OH_AudioSuite_Result insertAfter(
        const std::string &sourceNodeId, const std::string &targetNodeId, const Node &targetNode);
    
    //根据不同效果类型获取节点效果参数
    std::string GetOptionsByType(const Node& node);

    char *getNodeTypeName(OH_AudioNode_Type type);
};

#endif