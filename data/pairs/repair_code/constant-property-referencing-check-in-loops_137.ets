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