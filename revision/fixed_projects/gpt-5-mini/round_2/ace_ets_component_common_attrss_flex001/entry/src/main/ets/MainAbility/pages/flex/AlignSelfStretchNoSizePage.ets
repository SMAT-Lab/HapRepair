
/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import AttrsManager from '../../common/AttrsManager';
import router from '@ohos.router'

@Entry
@Component
struct AlignSelfStretchNoSizePage {
  @State _alignSelf: ItemAlign = ItemAlign.Auto;
  @State _wrap: FlexWrap = FlexWrap.NoWrap;
  @State _direction: FlexDirection = FlexDirection.Row;
  // 直接使用一般变量即可
  containerView: string = router.getParams() ? (router.getParams() as ESObject)['view']['containerView'] : ''
  targetView: string = router.getParams() ? (router.getParams() as ESObject)['view']['targetView'] : ''
  componentKey: string = router.getParams() ? (router.getParams() as ESObject)['view']['componentKey'] : ''
  parentComponentKey: string = router.getParams() ? (router.getParams() as ESObject)['view']['parentComponentKey'] : ''
  referenceComponentKey: string = router.getParams() ? (router.getParams() as ESObject)['view']['referenceComponentKey'] : ''
  parentWidth: number = router.getParams() ? (router.getParams() as ESObject)['view']['parentWidth'] : 0
  parentHeight: number = router.getParams() ? (router.getParams() as ESObject)['view']['parentHeight'] : 0
  compareWidth: number = router.getParams() ? (router.getParams() as ESObject)['view']['compareWidth'] : 0
  compareHeight: number = router.getParams() ? (router.getParams() as ESObject)['view']['compareHeight'] : 0

  onPageShow() {
    console.info('AlignSelfStretchNoSizePage page show called ')
    AttrsManager.registerDataChange((value: ESObject) => {
      this._alignSelf = value.alignSelf
      this._wrap = value.wrap
      this._direction = value.direction
    })
  }

  @Styles
  commonStyle(){
    .margin(0)
    .alignSelf(this._alignSelf)
    .key(this.componentKey)
  }

  @Builder
  buildTargetView() {
    if (this.targetView == 'AlphabetIndexer') {
      AlphabetIndexer({ arrayValue: ['#', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], selected: 0 }).commonStyle()
    } else if (this.targetView == 'Button') {
      Button('button1').commonStyle()
    } else if (this.targetView == 'Checkbox') {
      Checkbox({ name: 'checkbox1', group: 'checkboxGroup' }).commonStyle()
    } else if (this.targetView == 'CheckboxGroup') {
      CheckboxGroup({ group: 'checkboxGroup' }).commonStyle()
    } else if (this.targetView == 'DataPanel') {
      DataPanel({ values: [25], max: 100, type: DataPanelType.Circle }).commonStyle()
    } else if (this.targetView == 'DatePicker') {
      DatePicker({
        start: new Date('1970-1-1'),
        end: new Date('2100-1-1'),
        selected: new Date()
      }).commonStyle()
    } else if (this.targetView == 'Divider') {
      Divider().strokeWidth(8).vertical(true).commonStyle()
    } else if (this.targetView == 'Gauge') {
      Gauge({ value: 75 }).colors([[Color.White, 1], [Color.Black, 1]]).commonStyle()
    } else if (this.targetView == 'Image') {
      Image($r('app.media.icon')).commonStyle()
    } else if (this.targetView == 'ImageAnimator') {
      ImageAnimator().images([{ src: $r('app.media.icon') }]).commonStyle()
    } else if (this.targetView == 'LoadingProgress') {
      LoadingProgress().color(Color.Blue).commonStyle()
    } else if (this.targetView == 'Marquee') {
      Marquee({ start: true, step: 50, src: 'Running Marquee starts rolling' })
        .fontWeight(700).fontColor(Color.White).fontSize(48).commonStyle()
    } else if (this.targetView == 'Menu') {
      Menu() {
        MenuItem({ startIcon: $r('app.media.icon'), content: 'Menu' })
      }.commonStyle()
    } else if (this.targetView == 'Navigation') {
      Navigation().commonStyle()
    } else if (this.targetView == 'NavRouter') {
      NavRouter() {
        Row() {
          Image($r('app.media.icon'))
        }

        NavDestination() {
          Text('NavDestination').fontSize(10)
        }
      }.commonStyle()
    } else if (this.targetView == 'Progress') {
      Progress({ value: 50, type: ProgressType.Linear }).commonStyle()
    } else if (this.targetView == 'QRCode') {
      QRCode('QRCode').color(0xF7CE00).commonStyle()
    } else if (this.targetView == 'Radio') {
      Radio({ value: 'Radio1', group: 'radioGroup' }).commonStyle()
    } else if (this.targetView == 'Rating') {
      Rating({ rating: 4 }).stars(5).commonStyle()
    } else if (this.targetView == 'ScrollBar') {
      ScrollBar({ scroller: new Scroller(), state: BarState.On }) {
        Text().width(20).height(20).borderRadius(10)
      }.width(20).commonStyle()
    } else if (this.targetView == 'Stepper') {
      Stepper() {
        StepperItem() {
            Text('Page One')
        }
      }.commonStyle()
    } else if (this.targetView == 'Search') {
      Search({ placeholder: 'Type to search...' })
        .searchButton('SEARCH')
        .commonStyle()
    } else if (this.targetView == 'Select') {
      Select([{ value: 'aaa', icon: '/common/public_icon.svg' },
        { value: 'bbb', icon: '/common/public_icon.svg' }]).commonStyle()
    } else if (this.targetView == 'Slider') {
      Slider({ value: 50, style: SliderStyle.OutSet, direction: Axis.Vertical })
        .blockColor('#191970')
        .trackColor('#ADD8E6')
        .selectedColor('#4169E1')
        .commonStyle()
    } else if (this.targetView == 'Text') {
      Text('text').fontColor(Color.White).fontSize(30).commonStyle()
    } else if (this.targetView == 'TextArea') {
      TextArea({ placeholder: 'TextArea', }).commonStyle()
    } else if (this.targetView == 'TextClock') {
      TextClock({ timeZoneOffset: -8 }).format('hms').commonStyle()
    } else if (this.targetView == 'TextInput') {
      TextInput({ placeholder: 'TextInput' }).commonStyle()
    } else if (this.targetView == 'TextPicker') {
      TextPicker({ range: ['apple1', 'orange2', 'peach3'], selected: 1 }).commonStyle()
    } else if (this.targetView == 'TextTimer') {
      TextTimer({ count: 30000 }).commonStyle()
    } else if (this.targetView == 'TimePicker') {
      TimePicker().commonStyle()
    } else if (this.targetView == 'Toggle') {
      Toggle({ type: ToggleType.Switch }).commonStyle()
    } else if (this.targetView == 'Badge') {
      Badge({
        value: '',
        style: { badgeSize: 6, badgeColor: Color.Red }
      }) {
        Button('BadgeContent')
      }.commonStyle()
    } else if (this.targetView == 'Column') {
      Column() {
        Text('Column')
      }.commonStyle()
    } else if (this.targetView == 'Counter') {
      Counter() {
        Text('1')
      }.commonStyle()
    } else if (this.targetView == 'Flex') {
      Flex() {
        Text('Flex')
      }.commonStyle()
    } else if (this.targetView == 'GridRow') {
      GridRow({
        columns: 5,
        gutter: { x: 5, y: 10 },
        breakpoints: { value: ['400vp', '600vp', '800vp'], reference: BreakpointsReference.WindowSize },
        direction: GridRowDirection.Row
      }) {
        ForEach([Color.Red, Color.Orange, Color.Yellow, Color.Green,
          Color.Pink, Color.Grey, Color.Blue, Color.Brown], (color: Color) => {
          GridCol({ span: { xs: 1, sm: 2, md: 3, lg: 4 } }) {
            Row().width('100%').height('20vp')
          }.borderColor(color).borderWidth(2)
        })
      }.commonStyle()
    } else if (this.targetView == 'Grid') {
      Grid() {
        ForEach(['0', '1', '2', '3', '4'], () => {
          ForEach(['0', '1', '2', '3', '4'], (day: string) => {
            GridItem() {
              Text(day)
                .width('100%').height('100%').textAlign(TextAlign.Center)
            }
          })
        })
      }
      .columnsTemplate('1fr 1fr 1fr 1fr 1fr')
      .rowsTemplate('1fr 1fr 1fr 1fr 1fr')
      .commonStyle()
    } else if (this.targetView == 'List') {
      List({ space: 20, initialIndex: 0 }) {
        ForEach([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (item: number) => {
          ListItem() {
            Text('' + item)
              .width('100%')
              .textAlign(TextAlign.Center)
          }
        }, (item: number) => item.toString())
      }
      .listDirection(Axis.Vertical)
      .commonStyle()
    } else if (this.targetView == 'Navigator') {
      Navigator({ target: 'MainAbility/pages/index/Index' }) {
        Text('Navigator Go to index page')
          .width('100%').textAlign(TextAlign.Center)
      }.commonStyle()
    } else if (this.targetView == 'RelativeContainer') {
      RelativeContainer() {
        Row()
          .width(100)
          .height(100)
          .alignRules({
            top: { anchor: '__container__', align: VerticalAlign.Top },
            left: { anchor: '__container__', align: HorizontalAlign.Start }
          })
      }.commonStyle()
    } else if (this.targetView == 'Row') {
      Row() {
        Text('Row')
      }.commonStyle()
    } else if (this.targetView == 'Panel') {
      Panel(true) {
          Text('Panel')
      }
      .mode(PanelMode.Half)
      .commonStyle()
    } else if (this.targetView == 'Refresh') {
      Refresh({ refreshing: true }) {
        Text('Refresh').fontSize(30)
      }
      .commonStyle()
    } else if (this.targetView == 'Scroll') {
      Scroll() {
        Column() {
          ForEach([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (item: number) => {
            Text('Scroll' + item.toString())
              .width('90%')
              .height(10)
              .textAlign(TextAlign.Center)
          }, (item: number) => item.toString())
        }.width('100%')
      }
      .scrollBar(BarState.On)
      .commonStyle()
    } else if (this.targetView == 'SideBarContainer') {
      SideBarContainer(SideBarContainerType.Embed) {
        Column() {
          ForEach([1, 2, 3], (item: number) => {
            Column({ space: 5 }) {
              Image($r('app.media.icon')).width(64).height(64)
              Text('Index0' + item).fontSize(25)
            }
          })
        }.width('100%')
        .justifyContent(FlexAlign.SpaceEvenly)

          Text('SideBarContainer content').fontSize(25)
      }.commonStyle()
    } else if (this.targetView == 'Stack') {
      Stack() {
        Text('Stack').fontColor(Color.White)
      }.commonStyle()
    } else if (this.targetView == 'Swiper') {
      Swiper() {
        ForEach([1, 2, 3], (item: number) => {
          Text('Swiper' + item).width('90%').height(160)
        }, (item: number) => item.toString())
      }
      .indicator(true)
      .commonStyle()
    } else if (this.targetView == 'Tabs') {
      Tabs() {
        TabContent() {
          Column().width('100%').height('100%').backgroundColor('#00CB87')
        }
      }.commonStyle()
    } else if (this.targetView == 'WaterFlow') {
      WaterFlow() {
        ForEach([1, 2, 3, 4, 5, 6, 7, 8, 9], (item: number) => {
          FlowItem() {
            Column() {
              Text('WaterFlow' + item).fontSize(12).height('16')
              Image($r('app.media.icon')).width(50).height(50).objectFit(ImageFit.Fill)
            }
          }
        }, (item: number) => item.toString())
      }.commonStyle()
    } else if (this.targetView == 'Circle') {
      Circle({ width: 150, height: 150 }).commonStyle()
    } else if (this.targetView == 'Ellipse') {
      Ellipse({ width: 150, height: 80 }).commonStyle()
    } else if (this.targetView == 'Line') {
      Line()
        .startPoint([0, 0])
        .endPoint([50, 100])
        .stroke(Color.Black)
        .commonStyle()
    } else if (this.targetView == 'Polyline') {
      Polyline({ width: 100, height: 100 })
        .points([[0, 0], [20, 60], [100, 100]])
        .fillOpacity(0)
        .stroke(Color.Blue)
        .strokeWidth(3)
        .commonStyle()
    } else if (this.targetView == 'Polygon') {
      Polygon()
        .points([[50, 0], [0, 50], [20, 100], [80, 100], [100, 50]])
        .fill(Color.Red)
        .fillOpacity(0.6)
        .commonStyle()
    } else if (this.targetView == 'Path') {
      Path()
        .commands('M0 100 A30 20 20 0 0 200 100 Z')
        .fillOpacity(0)
        .stroke(Color.Black)
        .strokeWidth(3)
        .commonStyle()
    } else if (this.targetView == 'Rect') {
      Rect({ width: '90%', height: 80 })
        .radius(20)
        .fill(Color.Pink)
        .stroke(Color.Transparent)
        .commonStyle()
    } else if (this.targetView == 'Shape') {
      Shape() {
        Path().width(20).height(20).commands('M0 0 L400 0 L400 150 Z')
      }
      .viewPort({ x: -20, y: -5, width: 310, height: 90 })
      .fill(0x317AF7)
      .stroke(0xEE8443)
      .strokeWidth(10)
      .strokeLineJoin(LineJoinStyle.Miter)
      .strokeMiterLimit(5)
      .commonStyle()
    } else if (this.targetView == 'ColumnSplit') {
      ColumnSplit() {
        Text('1').width('100%').height(20).backgroundColor(0xF5DEB3).textAlign(TextAlign.Center)
      }
      .commonStyle()
    } else if (this.targetView == 'RowSplit') {
      RowSplit() {
        Text('1').width('10%').height(100).backgroundColor(0xF5DEB3).textAlign(TextAlign.Center)
      }.commonStyle()
    } else if (this.targetView == 'Blank') {
      Blank().commonStyle()
    }
  }

  build() {
    Column() {
      if (this.containerView == 'FlexNoReference') {
        Flex({ direction: this._direction, alignItems: ItemAlign.Start, wrap: this._wrap }) {
          this.buildTargetView();
        }
        .width(this.parentWidth)
        .height(this.parentHeight)
        .backgroundColor(Color.Green)
        .key(this.parentComponentKey)
      } else if (this.containerView == 'Flex') {
        Flex({ direction: this._direction, alignItems: ItemAlign.Start, wrap: this._wrap }) {
          this.buildTargetView();
          Button('referenceComponent')
            .width(this.compareWidth)
            .height(this.compareHeight)
            .key(this.referenceComponentKey)
        }
        .width(this.parentWidth)
        .height(this.parentHeight)
        .backgroundColor(Color.Green)
        .key(this.parentComponentKey)
      }
    }.backgroundColor(Color.Yellow)
  }
}