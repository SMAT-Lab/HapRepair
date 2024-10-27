import os

def parse_ets_file(file_path):
    description = []
    code_example = ""
    in_comment = True

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if in_comment and stripped_line.startswith('//'):
                description.append(stripped_line[2:].strip())
            else:
                in_comment = False
                code_example += stripped_line + " "  # 拼接成单行字符串

    return {
        'description': ' '.join(description),
        'code_example': code_example.strip()  # 去除末尾空白
    }

def build_rules_dict(directory):
    rules = {}
    for filename in os.listdir(directory):
        if filename.endswith('.ets'):
            rule_name = os.path.splitext(filename)[0]
            rule_key = f"@performance/{rule_name}"
            file_path = os.path.join(directory, filename)
            rule_data = parse_ets_file(file_path)
            rules[rule_key] = rule_data
    return rules

# 设置要遍历的目录路径
directory = 'pages'

# 生成字典
rules = build_rules_dict(directory)

# 打印结果
print("rules = {")
for rule, data in rules.items():
    print(f"    '{rule}': {{")
    print(f"        'description': '{data['description']}',")
    print(f"        'code_example': '{data['code_example']}'")
    print("    },")
print("}")
