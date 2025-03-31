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