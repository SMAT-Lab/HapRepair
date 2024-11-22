import os
import chardet

def convert_to_utf8(input_file, output_file):
    # 检测文件编码
    with open(input_file, 'rb') as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        original_encoding = detected['encoding']
        print(f"检测到的原始编码格式: {original_encoding}")

    # 读取文件并重新编码为 UTF-8
    with open(input_file, 'r', encoding=original_encoding) as f:
        content = f.read()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"文件已成功转换为 UTF-8 编码: {output_file}")

# 示例使用
input_dir = './mydefects/ets/pages/defects'
output_dir = './mydefects3/ets/pages/defects'
os.makedirs(output_dir, exist_ok=True)

for file in sorted(os.listdir(input_dir)):
    input_file = os.path.join(input_dir, file)
    output_file = os.path.join(output_dir, file)
    convert_to_utf8(input_file, output_file)
