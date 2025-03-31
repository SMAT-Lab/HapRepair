@Entry
@Component
struct App {
  build() {
    Column() {
      Scroll() {
        Flex({
          direction: FlexDirection.Column,
          justifyContent: FlexAlign.SpaceBetween,
          alignItems: ItemAlign.Start
        }) {
          ForEach(this.scenarioItems, (item: TestScenario, index: number) => {
            this.IngredientItem(item, index)
          }, (item: TestScenario, index: number) => item.name)
        }
      }
      .scrollBarWidth(20)
    }
    .height(ConfigData.WH_80_100)
    .padding({ top: 5, right: 10, left: 10 })
  }
}