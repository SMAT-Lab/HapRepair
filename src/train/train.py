import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import wandb

def init_distributed_mode():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.barrier()


# 在训练循环开始前初始化wandb
wandb.init(project="arktsLLM", name="202408012345")

# 加载预训练模型和分词器
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir="../model")

model.resize_token_embeddings(len(tokenizer))

# 配置LoRA
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)

init_distributed_mode()
model = DDP(model.to("cuda").half(), device_ids=[int(os.environ['LOCAL_RANK'])])  # 使用half precision (float16)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def load_train_jsonl_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                "file_path": item["file_path"],
                "content": item["content"]
            })
    return data

def load_test_jsonl_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                "file_path": item["file_path"],
                "prev": item["prev"],
                "target": item["target"]
            })
    return data

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']

        # 编码内容
        encoding = self.tokenizer(content, 
                                  truncation=True, 
                                  max_length=self.max_length, 
                                  padding='max_length', 
                                  return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            "file_path": item['file_path'],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # 使用输入作为标签，用于自回归训练
        }
    
class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prev_encoding = self.tokenizer(item["prev"], truncation=True, padding="max_length", max_length=self.max_length)
        target_encoding = self.tokenizer(item["target"], truncation=True, padding="max_length", max_length=self.max_length)
        
        return {
            "file_path": item["file_path"],
            "input_ids": torch.tensor(prev_encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(prev_encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(target_encoding["input_ids"], dtype=torch.long)
        }

# 修改数据加载部分
train_data = load_train_jsonl_dataset('processed_ets_files.jsonl')
test_data = load_test_jsonl_dataset('test_split.jsonl')

train_dataset = TrainDataset(train_data, tokenizer)
test_dataset = TestDataset(test_data, tokenizer)

# 修改 collate_fn 函数以适应不同的数据集结构
def train_collate_fn(batch):
    input_ids = torch.stack([ex["input_ids"] for ex in batch])
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch])
    labels = torch.stack([ex["labels"] for ex in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def test_collate_fn(batch):
    input_ids = torch.stack([ex["input_ids"] for ex in batch])
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch])
    labels = torch.stack([ex["labels"] for ex in batch])
    file_paths = [ex["file_path"] for ex in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "file_paths": file_paths}

# 更新 DataLoader
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=train_collate_fn, sampler=train_sampler, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=test_collate_fn, num_workers=4)

# training!
tb_writer = SummaryWriter(comment='data-parallel-training')
num_epochs = 5  # 总共训练5个epoch
save_path = "checkpoints"

os.makedirs(save_path, exist_ok=True)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step']

# 在训练开始前，如果有checkpoint，可以这样加载：
# checkpoint_path = f"{save_path}/temp_checkpoint.pt"
# if os.path.exists(checkpoint_path):
#     start_epoch, global_step = load_checkpoint(checkpoint_path, model, optimizer)
#     steps_per_epoch = len(train_dataloader)
#     start_step = global_step % steps_per_epoch
# else:
#     start_epoch, global_step, start_step = 0, 0, 0

start_epoch, global_step, start_step = 0, 0, 0

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_sampler.set_epoch(epoch)  # 确保在分布式训练中数据被正确打乱
    model.zero_grad()
    accumulation_steps = 4  # 累积梯度
    for step, batch in enumerate(train_dataloader):
        if epoch == start_epoch and step < start_step:
            continue  # 跳过已经处理过的步骤
       
        # 计算当前的global step
        global_step += 1

        input_ids = batch['input_ids'].to("cuda", dtype=torch.long)  # 确保索引张量使用long类型
        attention_mask = batch['attention_mask'].to("cuda", dtype=torch.float16)
        labels = batch['labels'].to("cuda", dtype=torch.long)

        # forward
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss.mean()  # 如果使用了DataParallel

        print(loss)

        # backward
        optimizer.zero_grad()
        loss = loss / accumulation_steps
        loss.backward()
        # 在这里添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        # 计算当前的round (全局步数)
        global_step = epoch * len(train_dataloader) + step

        # 打印进度
        if step % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {step}/{len(train_dataloader)}, Global Step {global_step}")

        # 每1000步保存一次
        if global_step % 1000 == 0 and global_step > 0:
            # 获取原始的 PEFT 模型
            peft_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

            # 获取 PEFT 状态字典
            peft_state_dict = get_peft_model_state_dict(peft_model)
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': peft_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, f"{save_path}/checkpoint_epoch{epoch}_step{step}.pt")

        if global_step % 100 == 0 and global_step > 0:
            # 获取原始的 PEFT 模型
            peft_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
            # 获取 PEFT 状态字典
            peft_state_dict = get_peft_model_state_dict(peft_model)
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': peft_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, f"{save_path}/temp_checkpoint.pt")

        wandb.log({"loss": loss.item()}, step=global_step)
        
        # 记录到TensorBoard
        tb_writer.add_scalar('loss', loss.item(), global_step)

    # 每个epoch结束后的操作
    print(f"Epoch {epoch+1} completed")

# 在训练结束后关闭wandb
wandb.finish()

tb_writer.close()
