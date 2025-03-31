import hilog from '@ohos.hilog';
@Entry
@Component
struct ControlPanel {
  build() {
    DatePicker()
      .onDateChange(() => {
        hilog.warn(8004, 'ControlPanel', 'onValueChange')
      })
  }
}