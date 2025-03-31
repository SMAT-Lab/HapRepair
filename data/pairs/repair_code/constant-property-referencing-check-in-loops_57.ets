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
