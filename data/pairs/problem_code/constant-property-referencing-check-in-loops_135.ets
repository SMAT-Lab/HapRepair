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