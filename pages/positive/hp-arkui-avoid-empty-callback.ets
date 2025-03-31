@Component
struct MyComponent {
  doSomething() {
    //业务逻辑
  }

  build() {
    Button('Click', { type: ButtonType.Normal, stateEffect: true })
      .onClick(() => {
        this.doSomething()
      })
  }
}