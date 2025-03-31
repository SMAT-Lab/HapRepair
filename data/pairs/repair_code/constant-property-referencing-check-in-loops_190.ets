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