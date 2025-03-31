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