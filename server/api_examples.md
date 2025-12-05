# API使用示例

## 使用codelinter进行代码缺陷检测

### 1. 检测单个代码文件

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "@Component\nstruct TestComponent {\n  @State private message: string = \"Hello World\"\n  \n  build() {\n    Column() {\n      Text(this.message)\n        .fontSize(20)\n        .fontWeight(FontWeight.Bold)\n    }\n    .width(\"100%\")\n    .height(\"100%\")\n  }\n}"
  }'
```

### 2. 检测整个harmony项目

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "project_path": "/path/to/your/harmony/project"
  }'
```

### 3. 使用特定规则检测

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "project_path": "/path/to/your/harmony/project",
    "detection_rules": ["hp-arkui-no-state-var-access-in-loop", "hp-performance-no-closures"]
  }'
```

## 响应格式示例

```json
{
  "success": true,
  "defects": [
    {
      "rule": "hp-arkui-no-state-var-access-in-loop",
      "type": "performance",
      "severity": "medium",
      "description": "State variable accessed in loop may cause performance issues",
      "file": "src/main/ets/pages/Index.ets",
      "line": 25,
      "column": 12
    }
  ],
  "total_defects": 1,
  "code_quality_score": 90.0,
  "message": "Found 1 defects in project in 2.34s"
}
```

## 修复代码

### 1. 修复检测到的缺陷

```bash
curl -X POST "http://localhost:8000/api/v1/repair" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "your_code_here",
    "repair_mode": "single",
    "use_rag": true,
    "model_name": "gpt-4o-mini"
  }'
```

### 2. 多轮修复

```bash
curl -X POST "http://localhost:8000/api/v1/repair" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "your_code_here",
    "repair_mode": "multi_round",
    "max_rounds": 3,
    "use_rag": true,
    "model_name": "gpt-4o-mini"
  }'
```

## 验证功能完整性

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "original_code": "original_code_here",
    "repaired_code": "repaired_code_here"
  }'
```

## 批量处理

### 1. 提交批量修复任务

```bash
curl -X POST "http://localhost:8000/api/v1/repair/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "codes": [
      "code_file_1_content",
      "code_file_2_content",
      "code_file_3_content"
    ],
    "model_name": "gpt-4o-mini"
  }'
```

### 2. 查询批量任务状态

```bash
curl -X GET "http://localhost:8000/api/v1/repair/batch/{task_id}"
```

## 健康检查

### 基础健康检查

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

### 深度健康检查

```bash
curl -X GET "http://localhost:8000/api/v1/health/deep"
```

## Python客户端示例

```python
import requests
import json

# 服务地址
BASE_URL = "http://localhost:8000/api/v1"

def detect_defects_in_project(project_path, rules=None):
    """检测harmony项目中的缺陷"""
    url = f"{BASE_URL}/detect"
    payload = {
        "project_path": project_path,
        "detection_rules": rules
    }
    
    response = requests.post(url, json=payload)
    return response.json()

def repair_code(code, model_name="gpt-4o-mini"):
    """修复代码缺陷"""
    url = f"{BASE_URL}/repair"
    payload = {
        "code": code,
        "repair_mode": "single",
        "use_rag": True,
        "model_name": model_name
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 检测项目缺陷
    result = detect_defects_in_project("/path/to/harmony/project")
    print(f"Found {result['total_defects']} defects")
    
    # 修复单个代码文件
    code = """
    @Component
    struct TestComponent {
      build() {
        Text('Hello')
      }
    }
    """
    
    repair_result = repair_code(code)
    if repair_result['success']:
        print("Code repaired successfully!")
        print(repair_result['repaired_code'])
```

## 环境配置

确保设置了必要的环境变量：

```bash
# .env文件示例
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key  # 可选，用于RAG功能
DEBUG=False
```

## 注意事项

1. **codelinter依赖**：确保系统中已安装codelinter工具
2. **项目路径**：使用绝对路径指定harmony项目目录
3. **权限**：确保服务有权限访问项目目录和执行codelinter命令
4. **超时**：大型项目检测可能需要几分钟时间
5. **并发**：避免同时对同一项目进行多次检测