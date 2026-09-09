# @performance/constant-property-referencing-check-in-loops

Static repair references: 17. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0136680167d9a44d`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}
function getNum(num: number): number {
  /* Year has (12 * 29 =) 348 days at least */
  let total: number = 348;
  for (let index: number = 0x8000; index > 0x8; index >>= 1) {
    // warning
    total += ((Time.info[num - Time.start] & index) !== 0) ? 1 : 0;
  }
  return total;
}
```

### Repair pattern

```arkts
class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}
function getNum(num: number): number {
  /* Year has (12 * 29 =) 348 days at least */
  let total: number = 348;
  const info = Time.info[num- Time.start];  
  for (let index: number = 0x8000; index > 0x8; index >>= 1) {
    if ((info & index) != 0) {
      total++;
    }
  }
  return total;
}
```

### Rationale

循环中一直使用Time.info[num - Time.start]，且该值在循环的过程中没有变化，应该放在循环外进行计算

## Example 2: `pair_047a8b6e8ff355a1`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Config {
  static settings: number[] = [5, 10, 15, 20, 25, 30];
}

function calculateSum(length: number): number {
  let sum: number = 0;
  let positionValue: number = 1;
  for (let i = 0; i < 5; i++) {
    sum += Config.settings[length - positionValue];
    positionValue++;
  }
  return sum;
}
```

### Repair pattern

```arkts
class Config {
  static settings: number[] = [5, 10, 15, 20, 25, 30];
}

function calculateSum(length: number): number {
  let sum: number = 0;
  for (let i = 0; i < 5; i++) {
    const index = length - (i + 1);  // 提取到循环外部
    sum += Config.settings[index];
  }
  return sum;
}
```

### Rationale

在每次循环迭代中计算 Config.settings[length - positionValue] 是不必要的，因为 length 和 positionValue 在每次迭代中不变。

## Example 3: `pair_0d18d055398189f9`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Constants {
  static values: number[] = [100, 200, 300, 400, 500];
}

function calculateDifference(start: number): number {
  let difference: number = 0;
  for (let k = 0; k < Constants.values.length; k++) {
    difference += Constants.values[k] - Constants.values[start];
  }
  return difference;
}
```

### Repair pattern

```arkts
class Constants {
  static values: number[] = [100, 200, 300, 400, 500];
}

function calculateDifference(start: number): number {
  let difference: number = 0;
  const baseValue = Constants.values[start]; // 提取到循环外
  for (let k = 0; k < Constants.values.length; k++) {
    difference += Constants.values[k] - baseValue;
  }
  return difference;
}
```

### Rationale

Constants.values[start] 在整个循环内都是不变的，因此不必每次都从数组中引用。

## Example 4: `pair_39c73a1e8683d9ca`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Data {
  static values: number[] = [7, 14, 21, 28];
}

function multiplyValues(multiplier: number): number {
  let product: number = 1;
  for (let n of Data.values) {
    product *= Data.values[multiplier] + n;  // 固定值重复使用
  }
  return product;
}
```

### Repair pattern

```arkts
class Data {
  static values: number[] = [7, 14, 21, 28];
}

function multiplyValues(multiplier: number): number {
  let product: number = 1;
  const multiplierValue = Data.values[multiplier];  // 提取到循环外
  for (let n of Data.values) {
    product *= multiplierValue + n;
  }
  return product;
}
```

### Rationale

Data.values[multiplier] 在循环中每次使用时不变，应提取到循环外。

## Example 5: `pair_4256540b0e5d1282`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Matrix {
  static dimensions: number[] = [5, 10, 15, 20];
}

function verifyDimension(target: number): boolean {
  for (let x = 0; x < Matrix.dimensions.length; x++) {
    if (Matrix.dimensions[target] === Matrix.dimensions[x]) { // 每次循环都访问相同值
      return true;
    }
  }
  return false;
}
```

### Repair pattern

```arkts
class Matrix {
  static dimensions: number[] = [5, 10, 15, 20];
}

function verifyDimension(target: number): boolean {
  const targetDimension = Matrix.dimensions[target]; // 提取到循环外
  for (let x = 0; x < Matrix.dimensions.length; x++) {
    if (targetDimension === Matrix.dimensions[x]) {
      return true;
    }
  }
  return false;
}
```

### Rationale

Matrix.dimensions[target] 在循环中不变，应在循环外提取。

## Example 6: `pair_556b0ce9466b45b2`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Settings {
  static components: string[] = ["ComponentA", "ComponentB", "ComponentC"];
}

function checkComponents(fixedIndex: number): boolean {
  for (let i = 0; i < Settings.components.length; i++) {
    if (Settings.components[i] === Settings.components[fixedIndex]) {  // 不必要的重复引用
      return true;
    }
  }
  return false;
}
```

### Repair pattern

```arkts
class Settings {
  static components: string[] = ["ComponentA", "ComponentB", "ComponentC"];
}

function checkComponents(fixedIndex: number): boolean {
  const targetComponent = Settings.components[fixedIndex]; // 提取到循环外
  for (let i = 0; i < Settings.components.length; i++) {
    if (Settings.components[i] === targetComponent) {
      return true;
    }
  }
  return false;
}
```

### Rationale

Settings.components[fixedIndex] 在迭代中是一个不变值，可以在循环开始之前读取。

## Example 7: `pair_63957f7e53561760`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Range {
  static bounds: number[] = [8, 16, 24, 32];
}

function detectBound(index: number): boolean {
  for (let b = 0; b < Range.bounds.length; b++) {
    if (Range.bounds[index] === Range.bounds[b]) { // 不变的值重复访问
      return true;
    }
  }
  return false;
}
```

### Repair pattern

```arkts
class Range {
  static bounds: number[] = [8, 16, 24, 32];
}

function detectBound(index: number): boolean {
  const checkBound = Range.bounds[index]; // 提取到循环外
  for (let b = 0; b < Range.bounds.length; b++) {
    if (checkBound === Range.bounds[b]) {
      return true;
    }
  }
  return false;
}
```

### Rationale

Range.bounds[index] 在循环中不变，应提取到循环外。

## Example 8: `pair_6e13b7f6843dfa19`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Metrics {
  static data: number[] = [3, 6, 9, 12];
}

function computeSum(factor: number): number {
  let sum: number = 0;
  for (let m = 0; m < Metrics.data.length; m++) {
    sum += Metrics.data[factor] + m; // 每次循环重复访问
  }
  return sum;
}
```

### Repair pattern

```arkts
class Metrics {
  static data: number[] = [3, 6, 9, 12];
}

function computeSum(factor: number): number {
  let sum: number = 0;
  const factorValue = Metrics.data[factor]; // 提取到循环外
  for (let m = 0; m < Metrics.data.length; m++) {
    sum += factorValue + m;
  }
  return sum;
}
```

### Rationale

Metrics.data[factor] 在循环中是固定值，应该在循环外提取。

## Example 9: `pair_7f60eb8ae1d36c36`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Catalog {
  static items: string[] = ['book', 'pen', 'notebook'];
}

function locateItem(searchIndex: number): boolean {
  for (let k = 0; k < Catalog.items.length; k++) {
    if (Catalog.items[k] === Catalog.items[searchIndex]) { // 不必要的重复访问
      return true;
    }
  }
  return false;
}
```

### Repair pattern

```arkts
class Catalog {
  static items: string[] = ['book', 'pen', 'notebook'];
}

function locateItem(searchIndex: number): boolean {
  const searchItem = Catalog.items[searchIndex]; // 提取到循环外
  for (let k = 0; k < Catalog.items.length; k++) {
    if (Catalog.items[k] === searchItem) {
      return true;
    }
  }
  return false;
}
```

### Rationale

Catalog.items[searchIndex] 在循环中多次使用且值不变，应提取到循环外。

## Example 10: `pair_80091bf698e953f5`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Values {
  static list: number[] = [100, 200, 300, 400];
}

function checkHighValue(ref: number): boolean {
  for (let v of Values.list) {
    if (Values.list[ref] < v) { // 无效的反复访问
      return false;
    }
  }
  return true;
}
```

### Repair pattern

```arkts
class Values {
  static list: number[] = [100, 200, 300, 400];
}

function checkHighValue(ref: number): boolean {
  const refValue = Values.list[ref]; // 提取到循环外
  for (let v of Values.list) {
    if (refValue < v) {
      return false;
    }
  }
  return true;
}
```

### Rationale

Values.list[ref] 在循环中每次使用时不变，应在循环外提取。

## Example 11: `pair_873277d8e03c355a`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Items {
  static collection: string[] = ['apple', 'banana', 'cherry'];
}

function findItem(targetIndex: number): boolean {
  for (let i = 0; i < Items.collection.length; i++) {
    if (Items.collection[i] === Items.collection[targetIndex]) { // 重复引用
      return true;
    }
  }
  return false;
}
```

### Repair pattern

```arkts
class Items {
  static collection: string[] = ['apple', 'banana', 'cherry'];
}

function findItem(targetIndex: number): boolean {
  const targetItem = Items.collection[targetIndex]; // 提取到循环外
  for (let i = 0; i < Items.collection.length; i++) {
    if (Items.collection[i] === targetItem) {
      return true;
    }
  }
  return false;
}
```

### Rationale

Items.collection[targetIndex] 在循环中多次使用且值不变，应提取到循环外。

## Example 12: `pair_88063b26151b55b3`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Database {
  static configurations: number[] = [22, 33, 44, 55, 66];
}

function updateSum(offset: number): number {
  let sum: number = 0;
  let key: number = 2;
  for (let i = 0; i < 4; i++) {
    sum += Database.configurations[key - offset];  // 多次访问相同值
  }
  return sum;
}
```

### Repair pattern

```arkts
class Database {
  static configurations: number[] = [22, 33, 44, 55, 66];
}

function updateSum(offset: number): number {
  let sum: number = 0;
  let key: number = 2;
  const configurationValue = Database.configurations[key - offset]; // 提取到循环外
  for (let i = 0; i < 4; i++) {
    sum += configurationValue;
  }
  return sum;
}
```

### Rationale

Database.configurations[key - offset] 在循环中每次迭代都访问相同的数组元素，这是不必要的。

## Example 13: `pair_88f055a0da3187bf`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Scores {
  static values: number[] = [90, 80, 85, 70, 95];
}

function calculateAverage(bonus: number): number {
  let total: number = 0;
  for (let j = 0; j < Scores.values.length; j++) {
    total += Scores.values[bonus] * j;  // 每次循环都访问相同值
  }
  return total / Scores.values.length;
}
```

### Repair pattern

```arkts
class Scores {
  static values: number[] = [90, 80, 85, 70, 95];
}

function calculateAverage(bonus: number): number {
  let total: number = 0;
  const bonusValue = Scores.values[bonus];  // 提取到循环外
  for (let j = 0; j < Scores.values.length; j++) {
    total += bonusValue * j;
  }
  return total / Scores.values.length;
}
```

### Rationale

Scores.values[bonus] 在循环中是不变的，应提取到循环外。

## Example 14: `pair_8f71d41708aaaa79`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Config {
  static limits: number[] = [10, 20, 30, 40];
}

function checkValidity(index: number): boolean {
  for (let value of Config.limits) {
    if (Config.limits[index] > value) {  // 不变值反复访问
      return false;
    }
  }
  return true;
}
```

### Repair pattern

```arkts
class Config {
  static limits: number[] = [10, 20, 30, 40];
}

function checkValidity(index: number): boolean {
  const limitThreshold = Config.limits[index];  // 提取到循环外
  for (let value of Config.limits) {
    if (limitThreshold > value) {
      return false;
    }
  }
  return true;
}
```

### Rationale

Config.limits[index] 在循环中每次使用都不变，应提取到循环外。

## Example 15: `pair_9bf65a084cd0dc15`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Collection {
  static entries: string[] = ['one', 'two', 'three'];
}

function isEntryMatch(refIndex: number): boolean {
  for (const entry of Collection.entries) {
    if (Collection.entries[refIndex] === entry) { // 固定值重复使用
      return true;
    }
  }
  return false;
}
```

### Repair pattern

```arkts
class Collection {
  static entries: string[] = ['one', 'two', 'three'];
}

function isEntryMatch(refIndex: number): boolean {
  const refEntry = Collection.entries[refIndex]; // 提取到循环外
  for (const entry of Collection.entries) {
    if (refEntry === entry) {
      return true;
    }
  }
  return false;
}
```

### Rationale

Collection.entries[refIndex] 在循环中不变，应提取到循环外。

## Example 16: `pair_e0d38396ec8b66a2`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Record {
  static data: number[] = [11, 12, 13, 14, 15];
}

function processValues(pointer: number, limit: number): number {
  let result: number = 1;
  for (let j = 0; j < limit; j++) {
      result *= Record.data[pointer - 1]; // 重复引用
  }
  return result;
}
```

### Repair pattern

```arkts
class Record {
  static data: number[] = [11, 12, 13, 14, 15];
}

function processValues(pointer: number, limit: number): number {
  let result: number = 1;
  const fixedValue = Record.data[pointer - 1]; // 提取到循环外部
  for (let j = 0; j < limit; j++) {
    result *= fixedValue;
  }
  return result;
}
```

### Rationale

Record.data[pointer - 1] 每次使用都固定不变，应提取到循环外。

## Example 17: `pair_fd253fbd327a41a8`

在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数

### Triggering pattern

```arkts
class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}
function getNum(num: number): number {
  /* Year has (12 * 29 =) 348 days at least */
  let total: number = 348;
  for (let year_0 = 1; year_0 <= 2024; year_0++) {
    // warning
    for(let month_0 = 1; month_0 <= 12; month_0++) {
      for(let day_0 = 1; day_0 <= 31; day_0++) {
        total += ((Time.info[num - Time.start] & day_0) !== 0) ? 1 : 0;
      }
    }

  }
  return total;
}
```

### Repair pattern

```arkts
class Time {
  static start: number = 0;
  static info: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
}
function getNum(num: number): number {
  /* Year has (12 * 29 =) 348 days at least */
  let total: number = 348;
  let time_info = Time.info[num - Time.start]
  for (let year_0 = 1; year_0 <= 2024; year_0++) {
    // warning
    for(let month_0 = 1; month_0 <= 12; month_0++) {
      for(let day_0 = 1; day_0 <= 31; day_0++) {
        total += ((time_info & day_0) !== 0) ? 1 : 0;
      }
    }

  }
  return total;
}
```

### Rationale

在三重循环中使用了常量Time.info[num - Time.start]，需要把常量提取出最外层循环的外面
