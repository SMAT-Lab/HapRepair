# @performance/typed-array-check

Static repair references: 15. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_29d57571316ef4d2`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const pixelValues: number[] = [255, 128, 64, 32, 16];
```

### Repair pattern

```arkts
const pixelValues = new Uint8ClampedArray([255, 128, 64, 32, 16]);
```

### Rationale

使用普通的 number[] 存储大量数值时效率低。建议使用Uint8ClampedArray。

## Example 2: `pair_2d89f4fd0240615f`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
let weights: number[] = Array(4).fill(2.5);
```

### Repair pattern

```arkts
let weights = new Float64Array(4); weights.fill(2.5);
```

### Rationale

使用普通 number[] 数组不利于性能。应替换为Float64Array。

## Example 3: `pair_2e55a26bbf88cdb5`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
let fibonacci: number[] = [0, 1, 1, 2, 3, 5, 8, 13, 21];
```

### Repair pattern

```arkts
let fibonacci = new Int32Array([0, 1, 1, 2, 3, 5, 8, 13, 21]);
```

### Rationale

普通的 number[] 数组可能导致性能问题。考虑使用Int32Array以优化处理效率。

## Example 4: `pair_30ce7a940fa868f1`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const typedArray1: number[] = new Array(1, 2, 3);
const typedArray2: number[] = new Array(4, 5, 6);
let res: number[] = new Array(3);
for (let i = 0; i < 3; i++) {
     res[i] = typedArray1[i] + typedArray2[i];
}
```

### Repair pattern

```arkts
const typedArray1 = new Int8Array([1, 2, 3]); 
const typedArray2 = new Int8Array([4, 5, 6]);  
let res = new Int8Array(3);
for (let i = 0; i < 3; i++) {
     res[i] = typedArray1[i] + typedArray2[i];
}
```

### Rationale

原代码使用普通的 number[] 数组来存储数值。这种做法在处理大量数值数据时效率较低，因为普通数组没有经过优化，性能可能较差。
应改用 TypedArray（如 Int8Array）来代替普通数组。TypedArray 是专门为数值数据设计的数据结构，它们在内存使用和性能方面都经过优化，适合高效处理大量数值数据。

## Example 5: `pair_48a508b8511686e6`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const matrix: number[][] = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];
const result: number[] = new Array(matrix.length);
for (let i = 0; i < matrix.length; i++) {
  result[i] = matrix[i][i];
}
```

### Repair pattern

```arkts
const matrix = [
  new Float64Array([1, 2, 3]),
  new Float64Array([4, 5, 6]),
  new Float64Array([7, 8, 9])
];
const result = new Float64Array(matrix.length);
for (let i = 0; i < matrix.length; i++) {
  result[i] = matrix[i][i];
}
```

### Rationale

原代码使用普通的 number[][] 数组来存储数值。这种做法在处理大量数值数据时效率较低。使用 Float64Array 可以提高内存使用效率和性能。

## Example 6: `pair_73ab87426dd2bede`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
let distances: number[] = [10, 20, 30, 40, 50];
```

### Repair pattern

```arkts
let distances = new Int32Array([10, 20, 30, 40, 50]);
```

### Rationale

使用 number[] 数组来存储数值会降低效率。应使用Int32Array提高性能。

## Example 7: `pair_9d0415a1ce93b8d4`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
let multipliers: number[] = Array(10).fill(1).map((_, index) => index * 1.5);
```

### Repair pattern

```arkts
let multipliers = new Float64Array(10); for (let i = 0; i < 10; i++) { multipliers[i] = i * 1.5; }
```

### Rationale

普通 number[] 数组对性能不利，可采用Float64Array以提升效率。

## Example 8: `pair_9ff10c4ea4bb9d4d`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
let vector: number[] = [3, 6, 9];
```

### Repair pattern

```arkts
let vector = new Int16Array([3, 6, 9]);
```

### Rationale

使用普通 number[] 数组会影响性能，可以使用Int16Array来优化。

## Example 9: `pair_c12df1b492079d66`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const squareArray: number[] = [1, 4, 9, 16, 25];
```

### Repair pattern

```arkts
const squareArray = new Uint8Array([1, 4, 9, 16, 25]);
```

### Rationale

使用普通的 number[] 数组来存储数值，效率较低。替换为Uint8Array以优化性能。

## Example 10: `pair_c4aeacfad536583d`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const arr1: number[] = [1, 2, 3, 4, 5];
const arr2: number[] = [6, 7, 8, 9, 10];
let sumArray: number[] = new Array(arr1.length);
for (let i = 0; i < arr1.length; i++) {
  sumArray[i] = arr1[i] + arr2[i];
}
```

### Repair pattern

```arkts
const arr1 = new Int8Array([1, 2, 3, 4, 5]);
const arr2 = new Int8Array([6, 7, 8, 9, 10]);
let sumArray = new Int8Array(arr1.length);
for (let i = 0; i < arr1.length; i++) {
    sumArray[i] = arr1[i] + arr2[i];
}
```

### Rationale

原代码使用普通的 number[] 数组来存储数值，这种做法在处理大量数值数据时效率较低，因为普通数组没有经过优化，性能可能较差。应改用 TypedArray（如 Int8Array）来代替普通数组。TypedArray 是专门为数值数据设计的数据结构，在内存使用和性能方面都经过优化，更适合高效处理大量数值数据。

## Example 11: `pair_d22207250e2bba79`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const dataPoints: number[] = Array.from({length: 5}, (_, i) => i * 10);
```

### Repair pattern

```arkts
const dataPoints = new Int16Array(5); for (let i = 0; i < 5; i++) { dataPoints[i] = i * 10; }
```

### Rationale

使用普通的 number[] 数组来存储数值，效率较低。可改用Int16Array以提高处理性能。

## Example 12: `pair_d39d8d5fc488fa3a`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const brightnessLevels: number[] = [100, 200, 150, 50];
```

### Repair pattern

```arkts
const brightnessLevels = new Uint8ClampedArray([100, 200, 150, 50]);
```

### Rationale

性能可能受到 number[] 数组的限制。使用Uint8ClampedArray可以提升效率。

## Example 13: `pair_d8645553e22dfcde`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
let bigNumbers: number[] = new Array(10000);
for (let i = 0; i < bigNumbers.length; i++) {
    bigNumbers[i] = i * 2;
}
```

### Repair pattern

```arkts
let bigNumbers = new Uint16Array(10000);
for (let i = 0; i < bigNumbers.length; i++) {
    bigNumbers[i] = i * 2;
}
```

### Rationale

原代码使用普通的 number[] 数组来存储大量的数值，这种做法在处理大量数值数据时效率较低，因为普通数组没有经过优化，性能可能较差。应改用 TypedArray（如 Uint16Array）来代替普通数组。TypedArray 是专门为数值数据设计的数据结构，在内存使用和性能方面都经过优化，更适合高效处理大量数值数据。

## Example 14: `pair_e96802438a9b8810`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
const temperatures: number[] = Array.from({length: 6}, (_, i) => i * 5 + 10);
```

### Repair pattern

```arkts
const temperatures = new Float32Array(6); for (let i = 0; i < 6; i++) { temperatures[i] = i * 5 + 10; }
```

### Rationale

普通 number[] 数组可能导致性能问题。建议改用Float32Array。

## Example 15: `pair_f85191474b6320b2`

数值数组推荐使用TypedArray。

### Triggering pattern

```arkts
let floatArray: number[] = [1.1, 2.2, 3.3, 4.4, 5.5];
let doubleArray: number[] = new Array(floatArray.length);
for (let i = 0; i < floatArray.length; i++) {
  doubleArray[i] = floatArray[i] * 2;
}
```

### Repair pattern

```arkts
let floatArray = new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5]);
let doubleArray = new Float32Array(floatArray.length);
for (let i = 0; i < floatArray.length; i++) {
    doubleArray[i] = floatArray[i] * 2;
}
```

### Rationale

原代码使用普通的 number[] 数组来存储数值，这种做法在处理大量数值数据时效率较低。通过使用 Float32Array 可以提高内存使用效率和性能。
