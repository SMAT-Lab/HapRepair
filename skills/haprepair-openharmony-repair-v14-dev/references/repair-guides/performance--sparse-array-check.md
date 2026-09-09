# @performance/sparse-array-check

Static repair references: 15. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_07a63f7322c61ff9`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let temps: number[] = new Array(50);
temps[49] = 22.5;
```

### Repair pattern

```arkts
let temps: number[] = [22.5];
```

### Rationale

数组长度为 50，但只在索引 49 赋值，稀疏数组导致空间浪费。

## Example 2: `pair_0fb1e75b080c6f52`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let count = 100000;
let result: number[] = new Array(count);
result = new Array();
result[9999] = 0;
```

### Repair pattern

```arkts
let index = 3;
let result: number[] = [];
result[index] = 0;
```

### Rationale

初始创建了一个长度为100000的数组 result，将数组重新初始化为空数组，然后在特定位置9999进行赋值。这种方式创建了一个稀疏数组，大部分元素仍然未定义，内存中保留了大量无效空间。
应当直接创建一个紧凑且仅包含需要元素的数组，确保内存使用高效。例如，在索引"3"处进行赋值，而不是在一个稀疏数组中进行赋值操作。

## Example 3: `pair_26090a8915104732`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let data: boolean[] = new Array(20000);
data[19999] = true;
```

### Repair pattern

```arkts
let data: boolean[] = [true];
```

### Rationale

创建了一个长度为 20000 的数组，但只在索引 19999 赋值，大部分元素未定义，浪费内存。

## Example 4: `pair_3bfe65171d3228cf`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let points: number[] = [];
points[1500] = 7;
```

### Repair pattern

```arkts
let points: number[] = [7];
```

### Rationale

在索引 1500 处赋值，其余位置未定义，造成内存浪费。应避免稀疏数组。

## Example 5: `pair_4e66b65593328831`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let stringArray: string[] = [];
stringArray[5000] = 'example';
```

### Repair pattern

```arkts
let stringArray: string[] = [];
stringArray[0] = 'example';
```

### Rationale

稀疏数组在索引 5000 处赋值，其余位置都未定义，导致内存浪费和性能问题。应直接创建一个包含需要元素的紧凑数组，避免稀疏数组的使用

## Example 6: `pair_600e6068874bac00`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let scores: number[] = [];
scores[999] = 50;
```

### Repair pattern

```arkts
let scores: number[] = [50];
```

### Rationale

在索引 999 处赋值，其余位置未定义，形成稀疏数组，导致内存浪费。

## Example 7: `pair_6bb7927824ea0c30`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let chars: string[] = new Array(500);
chars[499] = 'z';
```

### Repair pattern

```arkts
let chars: string[] = ['z'];
```

### Rationale

数组长度为 500，但只在索引 499 赋值，形成稀疏数组，浪费内存。

## Example 8: `pair_a26e3bfe2cd80d9e`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let items: string[] = new Array(3000);
items[2999] = 'item';
```

### Repair pattern

```arkts
let items: string[] = ['item'];
```

### Rationale

创建了数组长度为 3000，但仅在索引 2999 赋值，形成稀疏数组，性能下降。

## Example 9: `pair_a339127b4acedeba`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let colors: string[] = [];
colors[500] = 'blue';
```

### Repair pattern

```arkts
let colors: string[] = ['blue'];
```

### Rationale

在索引 500 处赋值，其他位置未定义，导致稀疏数组，性能不佳。

## Example 10: `pair_a969a2b77bd530e1`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let values: number[] = [];
values[3333] = 10;
```

### Repair pattern

```arkts
let values: number[] = [10];
```

### Rationale

仅在索引 3333 处赋值，形成稀疏数组，导致性能低下。应避免稀疏数组。

## Example 11: `pair_afeefb5b50d2b9c6`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let largeArray: number[] = new Array(1000);
largeArray[999] = 42;
```

### Repair pattern

```arkts
let largeArray: number[] = new Array();
largeArray.push(42);
```

### Rationale

创建了一个长度为 1000 的数组，但只在索引 999 处赋值。此时，数组大部分元素未定义，造成内存浪费。应直接创建一个包含所需元素的数组。

## Example 12: `pair_bb80dbe713c6ec85`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let flags: boolean[] = [];
flags[200] = false;
```

### Repair pattern

```arkts
let flags: boolean[] = [false];
```

### Rationale

在索引 200 处赋值，其余未定义，导致稀疏数组。应创建紧凑数组。

## Example 13: `pair_c276f0a129417332`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let numbers: number[] = [];
numbers[100] = 1;
numbers[1000] = 2;
numbers[10000] = 3;
```

### Repair pattern

```arkts
let numbers: number[] = [1, 2, 3];
```

### Rationale

稀疏数组中仅在索引 100、1000 和 10000 处赋值，其他位置均未定义，导致内存浪费和性能下降。应当创建一个紧凑的数组，避免稀疏数组的使用。

## Example 14: `pair_d6debf058f893b97`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let flags: boolean[] = [];
flags[1000] = true;
```

### Repair pattern

```arkts
let flags: boolean[] = [true];
```

### Rationale

仅在索引 1000 处赋值，形成稀疏数组，导致性能问题。

## Example 15: `pair_ee0777143087beeb`

建议避免使用稀疏数组。

### Triggering pattern

```arkts
let array: string[] = new Array(10005);
array[9998] = 'end';
```

### Repair pattern

```arkts
let array: string[] = ['end'];
```

### Rationale

创建了一个长度为10005的数组，并在索引9998处赋值，造成大部分元素未定义，内存中存在大量无效空间。应使用紧凑数组，避免稀疏数组。
