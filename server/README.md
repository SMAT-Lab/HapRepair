# ArkTS Code Repair Service

基于FastAPI的ArkTS代码缺陷修复服务，提供RESTful API接口用于代码缺陷检测、修复和验证。

## 功能特性

- 🔍 **专业代码缺陷检测** - 集成codelinter工具，支持完整的harmony项目检测
- 🔧 **智能代码修复** - 使用大语言模型进行代码修复
- ✅ **功能完整性验证** - 验证修复后代码的功能完整性
- 🚀 **批量处理** - 支持批量代码修复（异步处理）
- 🧠 **RAG增强** - 基于向量检索的修复建议生成
- 🔄 **多轮修复** - 支持多轮迭代修复
- 📊 **健康监控** - 完整的服务健康检查
- 🏗️ **项目级检测** - 直接在harmony项目目录运行codelinter命令

## API接口

### 健康检查
- `GET /api/v1/health` - 基础健康检查
- `GET /api/v1/health/deep` - 深度健康检查

### 代码处理
- `POST /api/v1/detect` - 检测代码缺陷
- `POST /api/v1/repair` - 修复代码缺陷
- `POST /api/v1/verify` - 验证代码功能完整性
- `POST /api/v1/repair/batch` - 批量修复代码
- `GET /api/v1/repair/batch/{task_id}` - 获取批处理状态

### RAG检索 (新增)
- `POST /api/v1/rag/search` - RAG检索相似代码模式
- `POST /api/v1/rag/similarity` - 代码相似度比较
- `POST /api/v1/rag/repair-suggestions` - 获取RAG修复建议
- `POST /api/v1/rag/embed` - 创建代码嵌入向量
- `GET /api/v1/rag/status` - RAG服务状态

## 快速开始

### 1. 环境配置

复制环境变量模板：
```bash
cp .env.example .env
```

编辑 `.env` 文件，配置必要的API密钥：
```bash
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

#### 开发模式
```bash
python main.py
```

#### 生产模式
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. 访问API文档

## 使用示例

### 代码缺陷检测

#### 检测整个harmony项目（推荐）
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "project_path": "/home/LLMCodeRepair/A21_C__open-harmony"
  }'
```

#### RAG
```bash
  curl -X POST "http://localhost:8000/api/v1/rag/extract-context" \
    -H "Content-Type: application/json" \
    -d '{
      "project_path": "/home/LLMCodeRepair/A21_C__open-harmony",
      "auto_detect": true,
      "context_window": 3
    }'
```

### 代码修复

还没做，直接调用LLM就行

### 验证
还没做，将修复后的代码写入，然后再调用CodeLinter，过滤掉错的修复(Error)以及Defect变多了的修复


## 配置说明

### 环境变量

| 变量名 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | 是 | - | OpenAI API密钥 |
| `PINECONE_API_KEY` | 否* | - | Pinecone API密钥（RAG功能需要） |
| `DEBUG` | 否 | False | 调试模式 |
| `PORT` | 否 | 8000 | 服务端口 |
| `MAX_REPAIR_ROUNDS` | 否 | 5 | 最大修复轮数 |
| `ENABLE_RAG` | 否 | True | 启用RAG功能 |

*注：如果启用RAG功能（`ENABLE_RAG=True`），则需要配置`PINECONE_API_KEY`

### 支持的模型

- OpenAI GPT系列：`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
- Deepseek系列：`deepseek-chat`, `deepseek-coder`

## 项目结构

```
server/
├── main.py              # FastAPI应用入口
├── api/                 # API路由模块
│   ├── health.py        # 健康检查接口
│   └── repair.py        # 代码修复接口
├── models/              # 数据模型
│   └── schemas.py       # Pydantic模型定义
├── services/            # 业务逻辑层
│   └── repair_service.py # 修复服务实现
├── config/              # 配置模块
│   └── settings.py      # 应用配置
├── requirements.txt     # Python依赖
├── Dockerfile          # Docker构建文件
├── docker-compose.yml  # Docker Compose配置
└── README.md           # 项目文档
```

## 监控和日志

### 健康检查

服务提供了两级健康检查：
- `/api/v1/health` - 基础健康状态
- `/api/v1/health/deep` - 详细的组件状态检查

### 日志配置

通过环境变量 `LOG_LEVEL` 和 `LOG_FILE` 配置日志级别和输出文件。

## 性能优化

### 建议配置

1. **生产环境**：使用多个worker进程
   ```bash
   uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
   ```

2. **GPU加速**：如果有GPU，配置CUDA环境以加速模型推理

3. **缓存**：考虑添加Redis缓存常用的修复结果

## 故障排除

### 常见问题

1. **导入错误**：确保项目根目录在Python路径中
2. **API密钥错误**：检查`.env`文件中的API密钥配置
3. **端口占用**：修改`PORT`环境变量使用其他端口
4. **内存不足**：对于大模型，确保有足够的内存

### 调试模式

设置 `DEBUG=True` 启用详细的错误信息和调试日志。

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

[License Type]