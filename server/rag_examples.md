# RAG API 使用示例

## RAG检索接口

基于检索增强生成(RAG)的代码修复建议系统，支持代码模式匹配和相似度搜索。

## API接口列表

### 1. RAG搜索 - 检索相似代码模式
`POST /api/v1/rag/search`

### 2. 代码相似度比较
`POST /api/v1/rag/similarity`

### 3. RAG服务状态
`GET /api/v1/rag/status`

### 4. 创建代码嵌入
`POST /api/v1/rag/embed`

### 5. 获取修复建议
`POST /api/v1/rag/repair-suggestions`

## 使用示例

### 1. RAG搜索相似代码模式

```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "@Component\nstruct TestComponent {\n  @State private message: string = \"hello\"\n  @State private unused: string = \"unused\"\n  \n  build() {\n    Text(this.message)\n  }\n}",
    "query": "remove unused state variables",
    "top_k": 5,
    "include_metadata": true
  }'
```

响应格式：
```json
{
  "success": true,
  "query_embedding_created": true,
  "matches": [
    {
      "id": "example_1",
      "score": 0.92,
      "rule": "hp-arkui-remove-redundant-state-var",
      "defect_type": "performance",
      "description": "Remove state variables that are not associated with a UI component",
      "original_code": "@State private unused: string = \"test\"",
      "fixed_code": "// Removed unused state variable"
    }
  ],
  "total_matches": 1,
  "search_time": 0.156,
  "message": "Found 1 similar patterns"
}
```

### 2. 代码相似度比较

```bash
curl -X POST "http://localhost:8000/api/v1/rag/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "source_code": "@State private message: string = \"hello\"",
    "target_codes": [
      "@State private text: string = \"world\"",
      "@State private count: number = 0",
      "let message = \"hello\""
    ],
    "similarity_threshold": 0.7
  }'
```

响应格式：
```json
{
  "success": true,
  "similarities": [
    {
      "index": 0,
      "target_code": "@State private text: string = \"world\"",
      "similarity": 0.85,
      "above_threshold": true
    },
    {
      "index": 1,
      "target_code": "@State private count: number = 0",
      "similarity": 0.75,
      "above_threshold": true
    },
    {
      "index": 2,
      "target_code": "let message = \"hello\"",
      "similarity": 0.65,
      "above_threshold": false
    }
  ],
  "most_similar": {
    "index": 0,
    "target_code": "@State private text: string = \"world\"",
    "similarity": 0.85,
    "above_threshold": true
  },
  "message": "Computed similarity for 3 code pairs"
}
```

### 3. 检查RAG服务状态

```bash
curl -X GET "http://localhost:8000/api/v1/rag/status"
```

响应格式：
```json
{
  "success": true,
  "rag_available": true,
  "vector_db_connected": true,
  "embedding_model_loaded": true,
  "index_info": {
    "total_vectors": 1500,
    "dimension": 1536,
    "index_name": "arkts-1536"
  },
  "message": "RAG service status retrieved"
}
```

### 4. 创建代码嵌入向量

```bash
curl -X POST "http://localhost:8000/api/v1/rag/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "@Component\nstruct MyComponent {\n  build() {\n    Text(\"Hello\")\n  }\n}"
  }'
```

响应格式：
```json
{
  "success": true,
  "embedding_created": true,
  "embedding_dimension": 1536,
  "message": "Embedding created successfully"
}
```

### 5. 获取RAG修复建议

```bash
curl -X POST "http://localhost:8000/api/v1/rag/repair-suggestions" \
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
  }'
```

响应格式：
```json
{
  "success": true,
  "suggestions": [
    {
      "confidence": 0.91,
      "rule": "hp-arkui-remove-redundant-state-var",
      "description": "Remove unused state variable",
      "original_code": "@State private unused: string = \"test\"",
      "fixed_code": "// Remove this line",
      "similarity_score": 0.91
    }
  ],
  "similar_patterns": [...],
  "confidence_score": 0.91,
  "message": "Generated 1 repair suggestions"
}
```

## Python客户端示例

```python
import requests
import json

class RAGClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def search_similar_patterns(self, code, query=None, top_k=5):
        """搜索相似的代码模式"""
        url = f"{self.base_url}/rag/search"
        payload = {
            "code": code,
            "query": query,
            "top_k": top_k,
            "include_metadata": True
        }
        
        response = requests.post(url, json=payload)
        return response.json()
    
    def compare_similarity(self, source_code, target_codes, threshold=0.7):
        """比较代码相似度"""
        url = f"{self.base_url}/rag/similarity"
        payload = {
            "source_code": source_code,
            "target_codes": target_codes,
            "similarity_threshold": threshold
        }
        
        response = requests.post(url, json=payload)
        return response.json()
    
    def get_repair_suggestions(self, code, defects, top_k=3):
        """获取RAG修复建议"""
        url = f"{self.base_url}/rag/repair-suggestions"
        payload = {
            "code": code,
            "defects": defects,
            "top_k": top_k
        }
        
        response = requests.post(url, json=payload)
        return response.json()
    
    def get_service_status(self):
        """获取RAG服务状态"""
        url = f"{self.base_url}/rag/status"
        response = requests.get(url)
        return response.json()

# 使用示例
if __name__ == "__main__":
    client = RAGClient()
    
    # 检查服务状态
    status = client.get_service_status()
    print(f"RAG Service Status: {status}")
    
    # 搜索相似模式
    code = """
    @Component
    struct TestComponent {
      @State private unused: string = "test"
      
      build() {
        Text("Hello")
      }
    }
    """
    
    patterns = client.search_similar_patterns(
        code=code,
        query="remove unused state variables",
        top_k=3
    )
    
    print(f"Found {patterns['total_matches']} similar patterns")
    for match in patterns['matches']:
        print(f"- Rule: {match['rule']}, Score: {match['score']:.2f}")
    
    # 获取修复建议
    defects = [{
        "rule": "hp-arkui-remove-redundant-state-var",
        "type": "performance",
        "severity": "low",
        "description": "Remove unused state variable"
    }]
    
    suggestions = client.get_repair_suggestions(code, defects)
    print(f"Repair suggestions: {len(suggestions['suggestions'])}")
```

## 注意事项

1. **依赖要求**：
   - RAG功能需要安装 `transformers` 和 `pinecone-client`
   - 嵌入模型会自动下载到 `MODEL_CACHE_DIR` 目录

2. **环境配置**：
   ```bash
   # 必需配置
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=arkts-1536
   
   # 可选配置
   MODEL_CACHE_DIR=/home/models
   EMBEDDING_MODEL=dunzhang/stella_en_1.5B_v5
   ```

3. **性能考虑**：
   - 首次启动时需要下载嵌入模型（~6GB）
   - 建议使用GPU加速模型推理
   - Pinecone查询通常在100-300ms内完成

4. **错误处理**：
   - 如果RAG依赖不可用，API会返回相应的错误信息
   - 服务会优雅降级，其他功能仍然可用

## 扩展功能

RAG服务支持以下扩展：
- 自定义嵌入模型
- 批量向量化
- 增量索引更新
- 多语言代码支持