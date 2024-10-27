# LLMCodeRepair

使用ArkTs语料微调大模型，并通过RAG进行propmt learning来进行代码缺陷修复

## 环境配置
```bash
conda create --name arktsLLM python=3.9 -y
conda activate arktsLLM
```

## 数据处理
请使用src/process_raw.ipynb脚本进行数据处理

## 模型训练
### 训练模型
```bash
cd src/train
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 模型量化
先通过merge_lora.ipynb脚本将模型合并，然后将tokenizer复制一份到merge后的模型文件夹中！！！
按照官网教程装好llama.cpp，然后执行以下命令(请使用另外的环境)
```bash
cd llama.cpp
python convert_hf_to_gguf.py ../OHAPP/merged/model/ --outtype f16 --outfile ../OHAPP/gguf/model.gguf

./llama-quantize ../OHAPP/gguf/model.gguf ../OHAPP/gguf/model_quant_8.gguf q8_0
```

下载安装好ollama，然后执行以下命令
```bash
cd OHAPP/gguf
ollama create arktsLLM -f ./Modelfile
```

其中，Modelfile内容如下:
```bash
FROM ./model_quant_8.gguf
```

## 代码修复RAG
### 向量数据库构建
请参考src/pinecone.ipynb脚本构建向量数据库以及查询

### 代码修复
请参考src/ollama.ipynb脚本进行代码修复