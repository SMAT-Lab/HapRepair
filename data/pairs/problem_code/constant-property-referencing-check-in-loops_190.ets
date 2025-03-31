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