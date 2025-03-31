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
