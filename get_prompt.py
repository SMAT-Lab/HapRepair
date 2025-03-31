import os
import time


def generate_fix_prompt(rag_prompt, code, sum_context, defect_description, error_location):
    # Simplified redundant container example
    redundant_container_example = """
Before:```arkts
Row() {
    Column() {  // ❌ Redundant container
        Text('Hello')
    }
}```

After:
```arkts
Row() {
    Text('Hello')  // ✅ Simplified structure
}
```
"""
    
    # Performance optimization example - made more focused
    fix_example = """
Before optimization:
```arkts
for(let i = 0; i < 100; i++) {
    if (items[2].value * 1.5 > 10) {  // ❌ Constant computation inside loop
        doSomething()
    }
}
```

After optimization:
```arkts
const threshold = items[2].value * 1.5  // ✅ Computed once before loop
for(let i = 0; i < 100; i++) {
    if (threshold > 10) {
        doSomething()
    }
}
```
"""

    # Difflib example - made more clear with annotations
    difflib_example = """
Original code:
```arkts
Column() {
    Column() {
        Text("Hello").fontSize(20)
    }
}
```

Difflib format:
```diff
Column() {
-   Column() {
        Text("Hello").fontSize(20)
-   }
}
```

The difflib shows:
• Lines with '-' will be removed
• Lines without markers stay unchanged
• Lines with '+' (if any) show new additions
"""

    variable_declaration_example = """
Variable Declaration Rules in struct:
```arkts
@Entry
@Component
struct MyComponent {
    // Valid declarations in struct scope:
    @State count: number = 0          // ✓ Decorator declarations
    @ObjectLink data: object = {}     // ✓ Decorator declarations
    private settings = new Settings() // ✓ private declarations
    options = { value: 0 }           // ✓ plain declarations
    
    // Invalid declarations in struct scope:
    let value = 0      // ❌ Cannot use let/const in struct scope
    const data = []    // ❌ Cannot use let/const in struct scope

    // Valid in regular functions:
    aboutToAppear() {
        let count = 0              // ✓ let/const allowed in functions
        const max = 100           // ✓ let/const allowed in functions
        for (let i = 0; i < 10; i++) {  // ✓ let allowed in loops
            this.data.push(i)
        }
    }

    // Invalid in UI components:
    build() {
        Row() {
            let value = 0  // ❌ Cannot declare variables in UI components
            Column() {
                const size = 100  // ❌ Cannot declare variables in UI components
            }
        }
        .onClick(() => {
            let count = 0  // ✓ let/const allowed in event handlers
            const max = 100  // ✓ let/const allowed in event handlers
        })
    }
}
```

```arkts
@Entry
@Component
struct MyComponent {
  build() {
    let irregularData: number[] = [];  // ❌ Cannot declare variables in UI components
    let layoutOptions: GridLayoutOptions = {  // ❌ Cannot declare variables in UI components
      regularSize: [1, 1],
      irregularIndexes: this.irregularData,
    };
    Grid(this.scroller, layoutOptions) {
      LazyForEach(this.groupDataSource, (item: LazyItem<UserFileDataItem>): void => {
        GridItem() {
          ImageGridItem()
        }
        .aspectRatio(1)
        .columnStart(item.get().index % this.gridRowCount)
        .columnEnd(item.get().index % this.gridRowCount)
      }, (item: LazyItem<AlbumDataItem>): string => item.getHashCode())
    }
  }
}
```

```arkts
@Entry
@Component
struct MyComponent {

  irregularData: number[] = [];  // ✓ declare variables 
  layoutOptions: GridLayoutOptions = {  //  declare variables
    regularSize: [1, 1],
    irregularIndexes: this.irregularData,
  };

  build() {
    Grid(this.scroller, this.layoutOptions) { // use variables
      LazyForEach(this.groupDataSource, (item: LazyItem<UserFileDataItem>): void => {
        GridItem() {
          ImageGridItem()
        }
        .aspectRatio(1)
        .columnStart(item.get().index % this.gridRowCount)
        .columnEnd(item.get().index % this.gridRowCount)
      }, (item: LazyItem<AlbumDataItem>): string => item.getHashCode())
    }
  }
}
```

"""

    discontinuous_code_example = """
Original segments:
```arkts

  @State totalPrice: number = 0;
...
  addItem(price: number) {
    this.totalPrice += price;
    this.totalPrice += this.calculateTax(price);
    this.totalPrice += this.calculateShipping(price);
  }
...
    Column() {
      Text('总价：' + this.totalPrice)
        .fontSize(18)
      Button('添加商品')
        .onClick(() => this.addItem(100))
    }
```

Repaired segments:
```arkts
  @State totalPrice: number = 0;
...
  addItem(price: number) {
    let total = this.totalPrice;
    total += price;
    total += this.calculateTax(price);
    total += this.calculateShipping(price);
    this.totalPrice = total;
  }
...
    Column() {
      Text('总价：' + this.totalPrice)
        .fontSize(18)
      Button('添加商品')
        .onClick(() => this.addItem(100))
    }
```

Difflib format:
```diff
  @State totalPrice: number = 0;
...
  addItem(price: number) {
-     this.totalPrice += price;
-     this.totalPrice += this.calculateTax(price);
-     this.totalPrice += this.calculateShipping(price);
+     let total = this.totalPrice;
+     total += price;
+     total += this.calculateTax(price);
+     total += this.calculateShipping(price);
+     this.totalPrice = total;
  }
...
    Column() {
      Text('总价：' + this.totalPrice)
        .fontSize(18)
      Button('添加商品')
        .onClick(() => this.addItem(100))
    }
```
"""

    complete_diff = """
Original code:
```arkts
List({ space: 12, initialIndex: 0 }) {
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
  })
}
```

Repair Result
```arkts
List({ space: 12, initialIndex: 0 }) {
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
  }, item: Item => item.title)
}
```

difflib format:
```diff
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
-  })
+  }, item => item.title)
```
"""

    list_example = """```arkts
List() {
  ......
}
.width(10)
.height(10)
```
"""

    # Main prompt - restructured for clarity
    prompt = f"""
Role: You are an ArkTS code fix assistant. Your task is to STRICTLY follow the fix patterns shown in the provided examples.

Key Rules:

1. Variable Declarations:
{variable_declaration_example}

2. Redundant Container Removal:
{redundant_container_example}

3. Performance Optimization:
{fix_example}

4. Code Change Format:
{difflib_example}

5. Handling Discontinuous Code:
{discontinuous_code_example}

6. Set key generator for ForEach:
{complete_diff}

7. WaterFlow Data Preload:
For @performance/waterflow-data-preload-check defects, preload logic MUST be added to FlowItem() components only:
```arkts
FlowItem()
.onAppear(() => ...
```

8. init-list-component:
For @hw-eslint/init-list-component defects, the width and height MUST be initialized after the List() component:
{list_example}

Reference Examples:
{rag_prompt}

IMPORTANT: You MUST:
1. Only apply fixes that match the patterns in the above examples
2. Do not introduce any fixes based on general programming knowledge
3. If a fix pattern is not found in the examples, do not attempt to fix that part

Input Analysis:
• Error Details: {defect_description}
• Error Location: {error_location}

Code Segments for Review:
The following shows discontinuous code segments related to the error. 
Lines marked with "..." indicate code that exists but is not shown:
```arkts
{sum_context}
```

Task:
1. Compare the error with the reference examples
2. If and ONLY if you find a matching fix pattern:
   • Apply that EXACT fix pattern
   • Do not modify the pattern or create variations
3. For any issues that don't match the examples:
   • Leave the code unchanged
   • Do not attempt creative fixes

Fix Process:
1. Identify which reference example matches your error
2. Follow that example's fix pattern EXACTLY
3. Only fix the visible code segments
4. Preserve "..." markers

Output Format:
You MUST return a complete difflib format that:
1. Shows ALL lines from the input segments, including unchanged lines
2. Preserves all "..." markers exactly as they appear in the input
3. Uses proper markers:
   • "-" prefix for lines to be removed
   • "+" prefix for lines to be added
   • No prefix for unchanged lines
4. Maintains the exact structure and flow of the input segments

Example Format:
Original segments with discontinuous markers:
{discontinuous_code_example}

Complete source code (for reference only):
```arkts
{code}
```

Note how:
• All original lines are present
• "..." markers are preserved
• Unchanged lines have no prefix
• Only modified lines have "+" or "-" prefixes
• The complete structure is maintained

CRITICAL: 
1. Your output MUST maintain the complete structure of the input segments
2. All lines from input must appear in output (with or without +/- prefix)
3. All "..." markers must be preserved
4. Only apply fixes that EXACTLY match the reference examples
5. Do not invent new fix patterns or apply general programming knowledge
6. The unchanged line should be kept unchanged !!! Don't add or remove any unchanged line
"""
    return prompt

def combine_repair_results(repair_results, code):
    final_fix_prompt = """
Task: EXACT Text Replacement Based on Difflib Markers ONLY

IMPORTANT - THIS IS A PURE TEXT OPERATION:
• You are a text replacement tool
• You ONLY process lines with "+" or "-" markers
• You MUST keep ALL other text EXACTLY as is
• No code understanding required or wanted

Here's the ONLY change pattern you should follow:

STARTING CODE:
```arkts
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
  })
```


```diff
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
-  })
+  }, item => item.title)
```

RESULT:
```arkts
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
  }, item => item.title)
```


EXACT Rules to Follow:
1. Lines with NO markers: MUST remain EXACTLY as they are
2. Lines with "-": MUST be REMOVED
3. Lines with "+": MUST be ADDED (without the "+")
4. SPACING and INDENTATION: MUST remain EXACTLY as in original
5. ALL OTHER CODE: MUST remain COMPLETELY UNCHANGED

CHANGES TO APPLY:
"""
    
    for context, res in repair_results:
        final_fix_prompt += f"""
SEGMENT TO MODIFY:
```arkts
{context}
```

DIFFLIB CHANGES(Given by gpt-o1 which may have the thinking of the change):
{res}

"""
        
    final_fix_prompt += f"""
Complete Source Code:
```arkts
{code}
```

YOUR EXACT STEPS:
1. Locate each ORIGINAL SEGMENT in the source code
2. For THAT SEGMENT ONLY:
   - REMOVE lines marked with "-"
   - ADD lines marked with "+" (without the "+")
   - Keep ALL other lines EXACTLY as they are
3. Do not touch ANY OTHER PART of the code
4. Preserve ALL spacing and indentation EXACTLY

This is a pure text replacement task:
• Treat it like a search-and-replace operation
• Only modify the exact text matches
• Preserve all spacing and indentation
• Make no other changes


⚠️ CRITICAL WARNINGS:
• You are a MECHANICAL text processor
• ONLY modify lines with "+" or "-" markers
• ALL OTHER LINES MUST REMAIN EXACTLY THE SAME
• NO code understanding or improvements allowed
• NO formatting changes allowed
• NO indentation changes allowed
• NO whitespace changes allowed
• EVERYTHING not marked with + or - MUST be identical

Return ONLY the complete source code with these exact replacements.
No explanations, no comments, just the processed code.
"""
    
    return final_fix_prompt

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # 使用最后一个隐藏层的平均池化作为句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

def get_rag_prompt(repair_example, model, tokenizer, index, rag_type, number=5):
    if number == 0:
        return ""

    query_text = repair_example["problem_code"]
    query_vector = get_embedding(query_text, model, tokenizer)
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            results = index.query(
                namespace="arkts",
                vector=query_vector.tolist(),
                top_k=number,
                include_metadata=True,
                filter={"rule": repair_example["rule"]}
            )
            break
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                raise e
            time.sleep(1)
    matches = results.matches
    if len(matches) == 0:
        return ""
    
    fix_prompt = ""
    for j, match in enumerate(matches):
        metadata = match['metadata']
        fix_prompt += (f"Demo {j+1}: \nRule Type: \n{metadata['rule']}\n\nDescription: \n{metadata['description']}\n\n"
                   f"Problem Code: \n```arkts\n{metadata['problem_code']}\n```\n\nFix Explanation: \n{metadata['problem_explain']}\n\n"
                   f"Fixed Code: \n\n```arkts\n{metadata['problem_fix']}\n```\n\n"
                   # f""" Following is the difflib results of repairing buggy code into fixed code:\n\n{metadata['difflib']}\n\n"""
                   # f"Following is the action to take to fix the buggy code into fixed code:\n\n{metadata['gpt_lib']}\n\n"
                )
        
        if rag_type == "gpt_diff":
            fix_prompt += f""" Following is the action to take to fix the buggy code into fixed code:\n\n{metadata['gpt_diff']}\n\n"""
        elif rag_type == "difflib":
            fix_prompt += f""" Following is the difflib results of repairing buggy code into fixed code:\n\n{metadata['difflib']}\n\n"""
    
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

def get_functionality_check_prompt(original_code, repaired_code, original_cfg=None, repaired_cfg=None):    
  prompt = f"""
You are a high-level code reviewer focusing on overall program functionality.

Your task is to determine if two versions of code (original and repaired) maintain the same core functionality and purpose, ignoring implementation details such as:
- Specific function implementations  
- Variable names and types
- Code structure and organization
- Control flow specifics
- Function call patterns
- Performance optimizations

Instead, focus on:
- The main purpose and objectives of the code
- Input/output behavior from an end-user perspective 
- Core business logic and requirements
- External behavior and interfaces
- Overall program workflow
- Control flow graph equivalence

For example:
- If both versions implement a user authentication system, verify they both achieve the core goal of authenticating users, regardless of how they implement it
- If both versions process data files, verify they produce equivalent results, even if they use different data structures or algorithms
- If both versions expose an API, verify the API provides the same capabilities, even if internal implementations differ

Note that the repaired code mainly focuses on performance optimizations, so you should:
- Verify the core functionality remains unchanged
- Ignore implementation differences and optimizations
- Check if the control flow graph remains equivalent
- Ensure the same features are still available

Original code:
{original_code}

Repaired code:
{repaired_code}

Original Control Flow Graph:
{original_cfg}

Repaired Control Flow Graph:
{repaired_cfg}

Please analyze if the repaired code maintains the same core functionality as the original code, ignoring implementation details.

Return your analysis in the following format:
{{
    "result": "success" | "failure", 
    "reason": "A detailed explanation focusing on whether the core functionality and purpose remain the same, not on implementation details"
}}
"""
  return prompt

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
    logger.getLogger().info("Building rules dictionary...")
    rules = {}
    for filename in os.listdir(negative_dir):
        if filename.endswith('.ets'):
            logger.getLogger().info(f"Processing rule file: {filename}")
            rule_name = os.path.splitext(filename)[0]
            rule_key = f"@rules/{rule_name}"
            negative_file_path = os.path.join(negative_dir, filename)
            rule_data = parse_ets_file(negative_file_path)

            positive_file_path = os.path.join(positive_dir, filename)
            if os.path.exists(positive_file_path):
                positive_example = get_positive_example(positive_file_path)
                rule_data['positive_code_example'] = positive_example
            else:
                logger.getLogger().warning(f"Positive example file not found: {filename}")
                rule_data['positive_code_example'] = None

            rules[rule_key] = rule_data
    logger.getLogger().info("Rules dictionary built")
    return rules

def deepseek_prompt(repair_results):
    final_fix_prompt = """
Task: EXACT Text Replacement Based on Difflib Markers ONLY

IMPORTANT - THIS IS A PURE TEXT OPERATION:
• You are a text replacement tool
• You ONLY process lines with "+" or "-" markers
• You MUST keep ALL other text EXACTLY as is
• No code understanding required or wanted

Here's the ONLY change pattern you should follow:

STARTING CODE:
```arkts
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
  })
```

```diff
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
-  })
+  }, item => item.title)
```

RESULT:
```arkts
  ForEach(FIRST_NAV_LIST, (item, index) => {
    ListItem() {
      ItemTemplate({ item: item })
    }
    .width('93.3%')
    .borderRadius(24)
    .padding({ left: '3.6%', right: '5.4%', top: 12, bottom: 12 })
    .backgroundColor('#ffffff')
  }, item => item.title)
```

EXACT Rules to Follow:

1. Lines with NO markers: MUST remain EXACTLY as they are
2. Lines with "-": MUST be REMOVED
3. Lines with "+": MUST be ADDED (without the "+")
4. SPACING and INDENTATION: MUST remain EXACTLY as in original

5. ALL OTHER CODE: MUST remain COMPLETELY UNCHANGED

CHANGES TO APPLY:
"""

    for context, res in repair_results:
        final_fix_prompt += f"""
SEGMENT TO MODIFY:
```arkts
{context}
```

DIFFLIB CHANGES:
{res}
"""
    final_fix_prompt += """
Complete Source Code:
```arkts
{code}
```
"""

    final_fix_prompt += """
YOUR EXACT STEPS:

1. Locate each ORIGINAL SEGMENT in the source code
2. For THAT SEGMENT ONLY:
   - REMOVE lines marked with "-"
   - ADD lines marked with "+" (without the "+")
   - Keep ALL other lines EXACTLY as they are
3. Do not touch ANY OTHER PART of the code
4. Preserve ALL spacing and indentation EXACTLY

This is a pure text replacement task:
• Treat it like a search-and-replace operation
• Only modify the exact text matches
• Preserve all spacing and indentation
• Make no other changes

⚠️ CRITICAL WARNINGS:
• You are a MECHANICAL text processor
• ONLY modify lines with "+" or "-" markers
• ALL OTHER LINES MUST REMAIN EXACTLY THE SAME
• NO code understanding or improvements allowed
• NO formatting changes allowed
• NO indentation changes allowed
• NO whitespace changes allowed
• EVERYTHING not marked with + or - MUST be identical

Return ONLY the complete source code with these exact replacements.
No explanations, no comments, just the processed code.
"""
    return final_fix_prompt


def parse_ets_file(file_path, logger=None):
    logger.info(f"Processing file: {file_path}")
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
    logger.getLogger().info(f"Getting positive example data: {file_path}")
    code_example = ""
    in_comment = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            code_example += stripped_line + " "
    return code_example.strip()

