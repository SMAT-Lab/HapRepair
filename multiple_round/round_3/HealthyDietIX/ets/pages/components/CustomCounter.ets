


@Preview
@Component
struct DecIcon {
  private click: () => void = undefined

  build() {
    Stack() {
      Image($r('app.media.icon_sub'))
        .width(20)
        .height(20)
        .objectFit(ImageFit.Contain)
    }
    .width(20)
    .height(20)
    .onClick(() => {
      this.click()
    })
  }
}

@Preview
@Component
struct IncIcon {
  private readonly click: () => void = undefined
  build() {
    Stack() {
      Image($r('app.media.icon_add'))
        .width(20)
        .height(20)
        .objectFit(ImageFit.Contain)
    }
    .width(20)
    .height(20)
    .onClick(() => {
      this.click()
    })
  }
}

@Component
export struct CustomCounter {
  @Prop value: string
  private onDec: () => void = undefined
  private onInc: () => void = undefined

  build() {
    Row() {
      DecIcon({ click: this.onDec })
      Text(this.value).margin({ left: 11, right: 11 })
      IncIcon({ click: this.onInc })
    }
  }
}

@Preview
@Entry
@Component
struct PreviewCustomCounter {
  @State weight: number = 50

  build() {
    Row() {
      CustomCounter({
        value: this.weight + 'g',
        onDec: () => {
          this.weight -= 50
        },
        onInc: () => {
          this.weight += 50
        }
      })
    }
  }
}