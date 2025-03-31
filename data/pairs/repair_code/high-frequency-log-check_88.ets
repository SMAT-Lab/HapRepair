@Entry
@Component
struct ControlPanel {
  build() {
    DatePicker()
      .onDateChange(() => {
        const ACTION = 'onValueChange';
      })
  }
}