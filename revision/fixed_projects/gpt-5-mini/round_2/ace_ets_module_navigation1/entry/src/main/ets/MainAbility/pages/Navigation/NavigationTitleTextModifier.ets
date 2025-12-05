import { LengthMetrics } from '@kit.ArkUI';
import { TextModifier } from '@ohos.arkui.modifier';

class MainTitleTextModifier extends TextModifier {
  applyNormalAttribute(instance: TextModifier): void {
      console.log(`testTag mainTitle use style1`);
      instance.fontColor('#FFFFC000')
      instance.fontSize(35)
      instance.fontWeight(FontWeight.Bolder)
      instance.fontStyle(FontStyle.Normal)
      instance.textShadow({radius: 5, offsetX: 9})
  }
}

class SubTitleTextModifier extends TextModifier {
  applyNormalAttribute(instance: TextModifier): void {
      console.log(`testTag subTitle use style1`);
      instance.fontColor('#FFFFCEEE')
      instance.fontSize(15)
      instance.fontWeight(FontWeight.Bolder)
      instance.fontStyle(FontStyle.Normal)
      instance.textShadow({radius: 5, offsetX: 9})
  }
}

@Entry
@Component
struct NavigationTitleTextModifier {
  private navPathStack: NavPathStack = new NavPathStack();
  // 初始化标题栏起始端内间距
  @State paddingStart: LengthMetrics = LengthMetrics.vp(10);
  // 初始化标题栏结束端内间距
  @State paddingEnd: LengthMetrics = LengthMetrics.vp(10);
  // 主标题样式修改器
  @State mainTitleModifier: MainTitleTextModifier = new MainTitleTextModifier();
  // 副标题样式修改器
  @State subTitleModifier: SubTitleTextModifier = new SubTitleTextModifier();
  @State applyModifier: boolean = false;

  build() {
    Navigation(this.navPathStack) {
      Column() {
        Row() {
          Text(`apply Modifier`)
          Toggle({isOn: this.applyModifier, type: ToggleType.Switch}).onChange((isOn: boolean) => {
            this.applyModifier = isOn;
          }).id('applyModifier_toggleId')
        }
        .padding({ top: 95, left: 5, right: 5 })
        .width(180)
        .justifyContent(FlexAlign.SpaceBetween)
        .padding({ top: 40, left: 5, right: 5 })
        .width(180)
        .justifyContent(FlexAlign.SpaceBetween)
      }
      .width('100%')
      .height('100%')
    }
    .titleMode(NavigationTitleMode.Full)
    .title(
      {main: 'Title', sub: 'subTitle'},
      this.applyModifier ?
        {
          paddingStart: this.paddingStart,
          paddingEnd: this.paddingEnd,
          mainTitleModifier: this.mainTitleModifier,
          subTitleModifier: this.subTitleModifier,
        } : {
        paddingStart: this.paddingStart,
        paddingEnd: this.paddingEnd
      })
  }
}