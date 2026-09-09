# @performance/number-init-check

Static repair references: 15. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0566cdc06aabca93`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let intNum = 1;
intNum = 1.1;
```

### Repair pattern

```arkts
let intNum = 1;
intNum = 2;
```

### Rationale

intNum被声明为int类型。应当避免将其转换成float类型

## Example 2: `pair_05f8e96e5febce55`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let halfValue: number = 0.5, bonus: number = 1.5; halfValue = 1; bonus = 2;
```

### Repair pattern

```arkts
let halfValue: number = 0.5, bonus: number = 1.5; halfValue = 0.8; bonus = 1.7;
```

### Rationale

halfValue被声明为float类型。应当避免将其转换成int类型

## Example 3: `pair_1489ce9009e2a42d`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let accuracy: number = 0.8;
let improvement: number = 2.0;

accuracy = 1;
improvement = 3;
```

### Repair pattern

```arkts
let accuracy: number;
accuracy = 0.8;
let improvement: number;
improvement = 2.0;

accuracy = 1.0;
improvement = 3;
```

### Rationale

accuracy被声明为float类型。应当避免将其转换成int类型

## Example 4: `pair_1e0ad284638ed028`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let height: number = 180, weight: number = 70; height = 175.5; weight = 68.8;
```

### Repair pattern

```arkts
let height: number = 180, weight: number = 70; height = 176; weight = 69;
```

### Rationale

height被声明为int类型。应当避免将其转换成float类型

## Example 5: `pair_445ffd4e3ddb269b`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let floatNum = 1.3;
floatNum = 2;
```

### Repair pattern

```arkts
let floatNum = 1.3;
floatNum = 2.4;
```

### Rationale

floatNum被声明为float类型。应当避免将其转换成int类型

## Example 6: `pair_511405a8eb53eb74`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let maxRate = 3.2; maxRate = 1;
```

### Repair pattern

```arkts
let maxRate = 3.2; maxRate = 2.8;
```

### Rationale

maxRate被声明为float类型。应当避免将其转换成int类型

## Example 7: `pair_526dd1ec34b713bc`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let floatNum1: number = 2.5, floatNum2: number = 1.5;
floatNum1 = 3;
floatNum2 = 4;
```

### Repair pattern

```arkts
let floatNum1: number = 2.5, floatNum2: number = 1.5;
floatNum1 = 3.5;
floatNum2 = 4.3;
```

### Rationale

floatNum被声明为float类型。应当避免将其转换成int类型

## Example 8: `pair_58ab370c2bab935a`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let minValue = 0; minValue = 0.5;
```

### Repair pattern

```arkts
let minValue = 0; minValue = 1;
```

### Rationale

minValue被声明为int类型。应当避免将其转换成float类型

## Example 9: `pair_7a4d182d7a823399`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let fullMarks: number = 10, grade: number = 5; fullMarks = 9.5; grade = 8.4;
```

### Repair pattern

```arkts
let fullMarks: number = 10, grade: number = 5; fullMarks = 9; grade = 8;
```

### Rationale

fullMarks被声明为int类型。应当避免将其转换成float类型

## Example 10: `pair_87a46d86fb293ea1`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let yearCount = 5; let average = 7.9; average = 5; yearCount = 5.5;
```

### Repair pattern

```arkts
let yearCount = 5; let average = 7.9; average = 6.3; yearCount = 4;
```

### Rationale

yearCount被声明为int类型。应当避免将其转换成float类型
average被声明为float类型。应当避免将其转换成int类型

## Example 11: `pair_a42563543b929b0e`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let intNum1: number = 2, intNum2: number = 1;
intNum1 = 3.5;
intNum2 = 4.3;
```

### Repair pattern

```arkts
let intNum1: number = 2, intNum2: number = 1;
intNum1 = 3;
intNum2 = 4;
```

### Rationale

intNum被声明为int类型。应当避免将其转换成float类型

## Example 12: `pair_aee98ca540a9d68f`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let speed = 15.0; speed = 20;
```

### Repair pattern

```arkts
let speed = 15.0; speed = 20.0;
```

### Rationale

speed被声明为float类型。应当避免将其转换成int类型

## Example 13: `pair_b760c12a1e3fbb56`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let width = 10; width = 5.6;
```

### Repair pattern

```arkts
let width = 10; width = 6;
```

### Rationale

width被声明为int类型。应当避免将其转换成float类型

## Example 14: `pair_c67d7474eea6550b`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let price = 99; let tax = 7.5; price = 89.9; tax = 8;
```

### Repair pattern

```arkts
let price: number;
price = 99;
let tax: number;
tax = 7.5;
price = 90;
tax = 7.0;
```

### Rationale

price被声明为int类型。应当避免将其转换成float类型
tax被声明为float类型。应当避免将其转换成int类型

## Example 15: `pair_e7f7d41b54fa4cca`

该规则将检查number是否正确使用。

### Triggering pattern

```arkts
let intNum = 3;
let floatNum = 2.5;
floatNum = 4; 
intNum = 1.8;
```

### Repair pattern

```arkts
let intNum = 3;
let floatNum = 2.5;
intNum = 4;
floatNum = 1.8;
```

### Rationale

intNum被声明为int类型。应当避免将其转换成float类型
floatNum被声明为float类型。应当避免将其转换成int类型
