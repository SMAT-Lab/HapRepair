#!/usr/bin/env python3
"""
测试脚本：测试完整的缺陷修复prompt生成功能
使用新的 /rag/generate-repair-prompts 端点
"""

import requests
import json
import time
import sys
import os

# 配置
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/api/v1/rag/generate-repair-prompts"

# 项目路径
PROJECT_PATH = "/home/LLMCodeRepair/A21_C__open-harmony"

def test_repair_prompts():
    """测试修复prompt生成功能"""
    print("=" * 80)
    print("🚀 测试缺陷修复Prompt生成功能")
    print("=" * 80)
    
    # 检查项目路径
    if not os.path.exists(PROJECT_PATH):
        print(f"❌ 项目路径不存在: {PROJECT_PATH}")
        return False
    
    print(f"✅ 项目路径: {PROJECT_PATH}")
    
    # 准备请求数据
    payload = {
        "project_path": PROJECT_PATH,
        "auto_detect": True
    }
    
    print(f"📤 请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print()
    
    try:
        # 发送请求
        print("🔄 正在生成修复prompt...")
        start_time = time.time()
        
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  处理时间: {elapsed_time:.2f}秒")
        print()
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            
            print("✅ 请求成功!")
            print("=" * 80)
            print("📊 结果摘要:")
            print(f"   ✅ 成功: {result.get('success', False)}")
            print(f"   📋 修复prompt数量: {result.get('total_prompts', 0)}")
            print(f"   🐛 总缺陷数量: {result.get('total_project_defects', 0)}")
            print(f"   📁 上下文组数量: {len(result.get('context_groups', []))}")
            print("=" * 80)
            
            # 显示修复prompts
            repair_prompts = result.get('repair_prompts', [])
            
            if repair_prompts:
                print("\n📝 生成的修复Prompts:")
                for i, rp in enumerate(repair_prompts, 1):
                    print(f"\n{'='*60}")
                    print(f"🎯 修复Prompt {i}:")
                    print(f"{'='*60}")
                    
                    # 显示缺陷组信息
                    defect_group = rp['defect_group']
                    print(f"📄 文件: {defect_group.get('file_path', '未知')}")
                    print(f"🐛 缺陷数量: {defect_group.get('defect_count', 0)}")
                    print(f"🔗 缺陷行: {defect_group.get('context_range', {}).get('defect_lines', [])}")
                    
                    # 显示规则
                    rules = rp.get('rules', [])
                    print(f"📋 规则: {', '.join(rules)}")
                    
                    # 显示相似示例数量
                    similar_count = len(rp.get('similar_examples', []))
                    print(f"📚 相似示例: {similar_count}个")
                    
                    # 显示prompt预览
                    prompt = rp.get('repair_prompt', '')
                    print(f"\n📝 Prompt预览 (前500字符):")
                    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                    
                    # 显示完整prompt长度
                    print(f"📏 Prompt长度: {rp.get('prompt_length', 0)}字符")
                    
                    # 保存单个prompt到文件
                    prompt_file = f"repair_prompt_{i}.md"
                    with open(prompt_file, 'w', encoding='utf-8') as f:
                        f.write(prompt)
                    print(f"💾 Prompt已保存到: {prompt_file}")
                    
                    print()
            else:
                print("⚠️  未生成修复prompts")
            
            # 保存完整结果
            output_file = "repair_prompts_complete.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 完整结果已保存到: {output_file}")
            
            return True
            
        else:
            print(f"❌ 请求失败!")
            print(f"   状态码: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时!")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败! 请确保服务器正在运行")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False

def test_with_sample_code():
    """使用示例代码测试"""
    print("\n" + "=" * 80)
    print("🧪 使用示例代码测试")
    print("=" * 80)
    
    # 示例代码
    sample_code = """@Component
struct LoginForm {
  @State username: string = ''
  @State password: string = ''
  @State message: string = ''
  
  build() {
    Column() {
      Text('登录')
      TextInput({placeholder: '用户名'})
      TextInput({placeholder: '密码', type: InputType.Password})
      Button('登录')
        .onClick(() => {
          // TODO: 处理登录逻辑
        })
    }
  }
}"""
    
    # 示例缺陷
    sample_defects = [
        {
            "rule": "performance/hp-arkui-remove-redundant-state-var",
            "line": 3,
            "message": "未使用的state变量",
            "file": "LoginForm.ets"
        },
        {
            "rule": "performance/hp-arkui-remove-unchanged-state-var",
            "line": 4,
            "message": "未变化的state变量",
            "file": "LoginForm.ets"
        }
    ]
    
    payload = {
        "code": sample_code,
        "defects": sample_defects,
        "context_window": 5
    }
    
    print("📤 使用示例代码和缺陷...")
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            repair_prompts = result.get('repair_prompts', [])
            
            if repair_prompts:
                print(f"✅ 生成 {len(repair_prompts)} 个修复prompt")
                
                # 显示第一个prompt
                first_prompt = repair_prompts[0]
                print(f"\n📝 示例修复Prompt:")
                print(first_prompt.get('repair_prompt', ''))
                
                # 保存示例
                with open("sample_repair_prompt.md", 'w', encoding='utf-8') as f:
                    f.write(first_prompt.get('repair_prompt', ''))
                print("💾 示例已保存到: sample_repair_prompt.md")
                
            return True
            
    except Exception as ex:
        print(f"❌ 示例测试失败: {ex}")
        return False

def check_server_health():
    """检查服务器健康状态"""
    try:
        health_url = f"{BASE_URL}/api/v1/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("🟢 服务器健康状态:")
            print(f"   状态: {health_data.get('status', 'unknown')}")
            return True
        else:
            print("🔴 服务器健康检查失败")
            return False
            
    except Exception as e:
        print("🔴 无法连接服务器")
        return False

def main():
    """主函数"""
    print("🚀 开始测试缺陷修复Prompt生成功能...")
    
    # 检查服务器
    if not check_server_health():
        sys.exit(1)
    
    # 测试1: 项目级检测
    print("\n1️⃣ 测试项目级检测...")
    project_success = test_repair_prompts()

if __name__ == "__main__":
    main()