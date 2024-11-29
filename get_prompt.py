import os


def generate_fix_prompt(rag_prompt, code, sum_context, defect_description, error_location):
    prompt = """
You are an AI debugging assistant. Your task is to fix the provided code based on the error log.
I will show you similar errors below. Please help me fix my code based on these error fixes.\n
{rag_content}
### Input:
Surrounding code of the defect:
{code_snippet}
Error log:
{error_log}
Error location:
{error_location}
### Task:
Please analyze the code and error log, then provide the fixed code directly.
You must take all the defects into account and fix them all.
Keep the original indentation of each code snippet.

### Requirements:
1. Extract all constant values and property references OUTSIDE ALL LOOPS to improve performance:
   - No constant values or properties should appear inside any loop, including nested loops.
   - Store all constants in variables BEFORE the outermost loop.
   - This rule applies to every level of nested loops.

2. Variable declarations must follow these rules:
   - Inside the `struct` or `class`:
     - Use `@State`, `private`, or direct declarations, such as:
       - `@State variable = value`
       - `private variable = value`
       - `variable = value`
     Example (Correct): `@State message = 'button'`
     Example (Wrong): `let message = 'button'`
   - Inside the `build()` method (which is nested in the `struct` or `class`):
     - Use `let variable = value` for temporary or local variables.
     Example (Correct): `let message = 'button'`
     Example (Wrong): `@State message = 'button'`

3. Do NOT declare variables inside UI elements within `build()`:
   - Variables needed by UI elements should either:
     - Be declared at the top of `build()` as `let`.
     - Or be declared in the `struct` or `class`.
   - Do NOT declare variables inside function with the deecoration `@Builder`

For example:
{declare_example}
     
4. Focus solely on fixing defects in the provided code. Avoid altering unrelated code or the overall structure.

MOST IMPORTANT: You MUST extract ANY constant property references and calculations OUTSIDE OF ALL LOOPS to improve performance. This means:
1. No constant values or properties should be referenced inside ANY loop (including nested loops)
2. All constant values must be stored in variables BEFORE the outermost loop
3. This applies to ALL levels of nested loops - constants cannot appear in ANY loop level
4. Even if a constant appears in the innermost loop of a 3+ level nested loop structure, it must be extracted outside the outermost loop
5. Just fix the defects in the surrounding code of the defect, don't change the code structure, don't change the irrelevant code.

For example:   
{example}

### Output format:
Return the explanation of how you fixed the code and then the fixed code snippets of the surrounding code of the defect.
"""

    declare_example = """
Following are the correct and wrong ways to declare variables:
Wrong:
```arkts
@Component
export struct AutoContentTable {
  private autoItemsX!: TestAuto[];
  private testItem!: TestData
  @State autoItems: TestAuto[] = [];
  // Wrong! Use `private` instead of `let` outside of UI element
  let localName: string = 'DaYuBlue';
  @Prop changeIndex: number;

  build() {
    // Wrong! Use `let` instead of `private` inside UI element
    private autoItems: TestAuto[] = [];
  }
}
```
Correct:
```arkts
@Component
export struct AutoContentTable {
  private autoItemsX!: TestAuto[];
  private testItem!: TestData
  @State autoItems: TestAuto[] = [];
  // Correct! Use `private` or nothing instead of `let` outside of UI element
  private localName: string = 'DaYuBlue';
  localName2: string = 'DaYuBlue';
  @Prop changeIndex: number;

  build() {
    // Correct! Use `let` or nothing instead of `private` inside UI element
    let localName: string = 'DaYuBlue';
    let localName2: string = 'DaYuBlue';
  }
}
```
"""
    fix_example = """
```arkts
let a: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let num: number = 2;
for(let i = 0; i < 100; i++) {
    for(let j = 0; j < 100; j++) {
        for(let k = 0; k < 100; k++) {
            if (a[num] % 2) == 0 {
                // do something
            }
        }
    }
}
```

You MUST extract the constant value a[num]outside of the loop to improve performance.
```arkts
let a: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let num: number = 2;
let temp: number = a[num];
for(let i = 0; i < 100; i++) {
    for(let j = 0; j < 100; j++) {
        for(let k = 0; k < 100; k++) {
            if (temp % 2) == 0 {
                // do something
            }
        }
    }
}
```
"""
    fix_prompt = prompt.format(rag_content=rag_prompt, entire_code=code, code_snippet=sum_context, error_log=defect_description, error_location=error_location, example=fix_example, declare_example=declare_example)
    return fix_prompt

def combine_repair_results(repair_results, code):
    final_fix_prompt = "Following are the surrounding code of the defect and the fixed code:\n"
    for context, res in repair_results:
        final_fix_prompt += f"Surrounding code of the defect:\n```arkts\n{context}\n```\nFixed code:\n```arkts\n{res}\n```\n"
        
    
    final_fix_prompt += f"""
Please combine all the fixed code snippets to get the final fixed code of the entire file. 
Don't change the code structure, just fix the defects and don't change the irrelevant code.
The entire file is as follows:\n```arkts\n{code}\n```\nReturn the final fixed code directly.

Requirements:
1. Keep the original indentation of each code snippet.
2. Keep the original code structure.
3. Ensure that the final fixed code is syntactically correct and can be compiled.(Parentheses, brackets, etc. should be matched correctly)
    """

    return final_fix_prompt

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # 使用最后一个隐藏层的平均池化作为句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

def get_rag_prompt(repair_example, model, tokenizer, index, number=5):
    query_text = repair_example["problem_code"]
    query_vector = get_embedding(query_text, model, tokenizer)
    results = index.query(
        namespace="arkts",
        vector=query_vector.tolist(),
        top_k=number,
        include_metadata=True,
        filter={"rule": repair_example["rule"]}
    )
    matches = results.matches
    if len(matches) == 0:
        return ""
    
    fix_prompt = ""
    for j, match in enumerate(matches):
        metadata = match['metadata']
        fix_prompt += (f"Demo {j+1}: \nRule Type: \n{metadata['rule']}\n\nDescription: \n{metadata['description']}\n\n"
                   f"Problem Code: \n```arkts\n{metadata['problem_code']}\n```\n\nFix Explanation: \n{metadata['problem_explain']}\n\n"
                   f"Fixed Code: \n\n```arkts\n{metadata['problem_fix']}\n```\n\n"
                   f"Following is the action to take to fix the buggy code into fixed code:\n\n{metadata['diff']}\n\n"
                )
    
    return fix_prompt


def get_context_extraction_prompt():
    negative_directory = './pages/negative'
    positive_directory = './pages/positive'
    rules = build_rules_dict(negative_directory, positive_directory)

    system_prompt = """
    I am an expert at analyzing CodeLinter results and extracting minimal code blocks containing defects. When you provide me with ArkTS code and CodeLinter analysis results, I will:

    1. Learn the defect patterns from the rules
    2. Take the CodeLinter results as ground truth defect locations
    3. For each defect reported by CodeLinter:
       - Find the minimal code block containing the defect and its required context
       - Include any related code that is necessary to understand the defect
       - Ensure the extracted block captures the full scope of the issue
      

    Input format:
    1. Complete ArkTS source code
    2. CodeLinter results in JSON:
    [
      {
        "rule": "@performance/rule-name",
        "line": line_number, 
        "message": "Issue description",
        "code": "Code snippet"
      },
      {
        "rule": "@performance/rule-name",
        "line": line_number, 
        "message": "Issue description",
        "code": "Code snippet"
      }
    ]

    Important notes:
    - I will treat each CodeLinter result as definitive evidence of a defect
    - The defect_block will be the complete code unit containing the issue, for example, the define and use of a variable. 
    - You should find all the code related to the defect and return the defect_block. For example, for a constant-property-referencing-check-in-loops defect, you need to extract the constant variable assignment outside of the outermost loop
    - Line numbers will be preserved for accurate block replacement
    - Output will be valid JSON only, sorted by line number
    - Each block must be independently understandable and fixable

    I must output valid JSON only, sorted by line number. Make sure the output is correct JSON format which can be parsed by json.loads() and follow the format: 
    ```json
    [
      {
        "rule": "@performance/rule-name",
        "defect_block": {
          "code": ["Code block containing the defect"],
          "start_line": "first_line_number",
          "end_line": "last_line_number",
          "line of interest": "The code where the defect is reported by CodeLinter. Code instead of line number!!"
        },
        "analysis": "The description of the defect"
      },
      ...
    ]
    ```

    Following are the defect patterns I will look for:
    """

    for rule, details in rules.items():
        system_prompt += "Rule: {}\nDescription: {}\nDefect Example:\n{}\nFixed Example:\n{}\n\n".format(
            rule,
            details['description'],
            details['defect_code_example'],
            details['positive_code_example']
        )

    return system_prompt

def get_defect_extraction_prompt():
    negative_directory = './pages/negative'
    positive_directory = './pages/positive'
    rules = build_rules_dict(negative_directory, positive_directory)

    system_prompt = f"""
I am a code analyzer specialized in detecting performance defects in ArkTS code. I will carefully analyze the code line by line from top to bottom to identify any performance issues based on the rules you provided.

When you give me an ArkTS file, I will:
1. Read through the code sequentially from the first line to the last line
2. For each line, check if it violates any of the performance rules
3. If a defect is found, I will record:
   - The specific rule that was violated
   - The rule's description
   - The line number where the defect occurs
   - The problematic code snippet
   - Why this code violates the rule based on the rule's description

Please provide the code you want me to analyze. I will return the results in the following JSON format:

[
  {{
    "rule": "@performance/hp-arkui-use-reusable-component",
    "description": "避免在for、while等循环逻辑中频繁读取状态变量。通用丢帧场景下，建议优先修改。", 
    "line": 5,
    "defect_snippet": "<problematic_code>"
  }}
]

Notes:
- Results will be ordered by line number (ascending)!!!!
- Code snippets will use \\n for newlines
- I will detect ALL defects in the code, not just the first one found
- If no defects are found, I will return an empty array []
- I will only output valid JSON without any additional text

Here are the rules and their descriptions:
"""

    for rule, details in rules.items():
        system_prompt += "Rule: {}\nDescription: {}\nPositive Example Without Defect:\n{}\nNegative Example With Defect:\n{}\n\n".format(
            rule,
            details['description'],
            details['positive_code_example'],
            details['defect_code_example']
        )

    return system_prompt



def judge_need_context_prompt():
    
    negative_directory = './pages/negative'
    positive_directory = './pages/positive'
    rules = build_rules_dict(negative_directory, positive_directory)
    system_prompt = """
    Your task is to judge whether the repair process should consider the context or not(can just repair the defect in one line, or should repair in multiplt lines). 
    Following are the defects:
    """
    for rule, details in rules.items():
        system_prompt += "Rule: {}\nDescription: {}\nPositive Example Without Defect:\n{}\nNegative Example With Defect:\n{}\n\n".format(
            rule,
            details['description'],
            details['positive_code_example'],
            details['defect_code_example']
        )

    system_prompt += "You should output an array containing the defect's name which should get the context to fix the defect in multiple lines"

    return system_prompt

def build_rules_dict(negative_dir, positive_dir, logger=None):
    logger.getLogger().info("开始构建规则字典...")
    rules = {}
    for filename in os.listdir(negative_dir):
        if filename.endswith('.ets'):
            logger.getLogger().info(f"处理规则文件: {filename}")
            rule_name = os.path.splitext(filename)[0]
            rule_key = f"@performance/{rule_name}"
            negative_file_path = os.path.join(negative_dir, filename)
            rule_data = parse_ets_file(negative_file_path)

            positive_file_path = os.path.join(positive_dir, filename)
            if os.path.exists(positive_file_path):
                positive_example = get_positive_example(positive_file_path)
                rule_data['positive_code_example'] = positive_example
            else:
                logger.getLogger().warning(f"未找到对应的正例文件: {filename}")
                rule_data['positive_code_example'] = None

            rules[rule_key] = rule_data
    logger.getLogger().info("规则字典构建完成")
    return rules



def parse_ets_file(file_path, logger=None):
    logger.info(f"解析文件: {file_path}")
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
                code_example += stripped_line + " "

    return {
        'description': ' '.join(description),
        'defect_code_example': code_example.strip()
    }

def get_positive_example(file_path, logger=None):
    logger.getLogger().info(f"获取正例数据: {file_path}")
    code_example = ""
    in_comment = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            code_example += stripped_line + " "
    return code_example.strip()