#!/bin/bash

# RAG API 的 curl 调用示例

BASE_URL="http://localhost:8000/api/v1"

echo "🚀 RAG API curl 调用示例"
echo "================================"

# 1. 检查RAG服务状态
echo "1. 检查RAG服务状态"
echo "curl -X GET \"$BASE_URL/rag/status\""
curl -X GET "$BASE_URL/rag/status" | jq '.'
echo -e "\n"

# 2. 创建代码嵌入向量
echo "2. 创建代码嵌入向量"
echo "curl -X POST \"$BASE_URL/rag/embed\""
curl -X POST "$BASE_URL/rag/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "@Component\nstruct TestComponent {\n  @State private message: string = \"hello\"\n  \n  build() {\n    Text(this.message)\n  }\n}"
  }' | jq '.'
echo -e "\n"

# 3. RAG搜索相似代码模式
echo "3. RAG搜索相似代码模式"
echo "curl -X POST \"$BASE_URL/rag/search\""
curl -X POST "$BASE_URL/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "@Component\nstruct TestComponent {\n  @State private message: string = \"hello\"\n  @State private unused: string = \"unused\"\n  \n  build() {\n    Text(this.message)\n  }\n}",
    "query": "remove unused state variables",
    "top_k": 5,
    "include_metadata": true
  }' | jq '.'
echo -e "\n"

# 4. 代码相似度比较
echo "4. 代码相似度比较"
echo "curl -X POST \"$BASE_URL/rag/similarity\""
curl -X POST "$BASE_URL/rag/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "source_code": "@State private message: string = \"hello\"",
    "target_codes": [
      "@State private text: string = \"world\"",
      "@State private count: number = 0",
      "let message = \"hello\""
    ],
    "similarity_threshold": 0.7
  }' | jq '.'
echo -e "\n"

# 5. 获取RAG修复建议
echo "5. 获取RAG修复建议"
echo "curl -X POST \"$BASE_URL/rag/repair-suggestions\""
curl -X POST "$BASE_URL/rag/repair-suggestions" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "@Component\nstruct TestComponent {\n  @State private unused: string = \"test\"\n  \n  build() {\n    Text(\"Hello\")\n  }\n}",
    "defects": [
      {
        "rule": "hp-arkui-remove-redundant-state-var",
        "type": "performance",
        "severity": "low",
        "description": "Remove state variables that are not associated with a UI component"
      }
    ],
    "top_k": 3
  }' | jq '.'
echo -e "\n"

echo "✅ 所有RAG API示例完成！"