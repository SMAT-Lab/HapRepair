/**
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
import matrix4 from '@ohos.matrix4'

let matrix = matrix4.init([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

@Entry
@Component
struct comSizeAbilityTest_07 {
  select: boolean = true
  public iconStr: ResourceStr = $r("app.media.icon")
  public iconStr2: ResourceStr = $r("app.media.icon")

  @Builder
  MyMenu1() {
    Column() {
      Menu() {
        MenuItemGroup() {
          MenuItem({ content: "菜单选项" })
        }
        .backgroundColor('gray')
        .key("MenuItemGroup_Size")
        .size({ width: 90, height: 90 })
        .translate({ x: 1, y: 1, z: 1 })
        .scale({
          x: 1,
          y: 1,
          z: 1,
          centerX: 1,
          centerY: 1
        })
        .rotate({
          x: 1,
          y: 1,
          z: 1,
          angle: 0,
          centerX: 1,
          centerY: 1
        })
        .transform(matrix)
      }.height(90).width(90)
    }
  }

  @Builder
  MyMenu2() {
    Column() {
      Menu() {
        MenuItem({
          content: "Item",
        })
          .backgroundColor('gray')
          .key("MenuItem_Size")
          .size({ width: 90, height: 90 })
          .translate({ x: 1, y: 1, z: 1 })
          .scale({
            x: 1,
            y: 1,
            z: 1,
            centerX: 1,
            centerY: 1
          })
          .rotate({
            x: 1,
            y: 1,
            z: 1,
            angle: 0,
            centerX: 1,
            centerY: 1
          })
          .transform(matrix)
      }
    }
  }

  @Builder
  MyMenu3() {
    Column() {
      Menu() {
        MenuItem({
          startIcon: $r("app.media.icon"),
          content: "Menu",
          endIcon: $r("app.media.icon"),
        })
      }
      .backgroundColor('gray')
      .key("Menu_Size")
      .size({ width: 100, height: 100 })
      .translate({ x: 1, y: 1, z: 1 })
      .scale({
        x: 1,
        y: 1,
        z: 1,
        centerX: 1,
        centerY: 1
      })
      .rotate({
        x: 1,
        y: 1,
        z: 1,
        angle: 0,
        centerX: 1,
        centerY: 1
      })
      .transform(matrix)
    }
  }

  build() {
    Scroll() {
      Column({ space: 20 }) {
        Column() {
          Button(`ShowSize： MenuItemGroup`)
            .fontSize(16)
            .margin(5)
            .key('btn_MenuItemGroup')
            .bindMenu(this.MyMenu1, {
              showInSubWindow: false
            })
        }

        Column() {
          Button(`ShowSize: MenuItem`).fontSize(16).margin(5)
            .key('btn_MenuItem')
            .bindMenu(this.MyMenu2, {
              showInSubWindow: false
            })
        }

        Column() {
          Button(`ShowSize: Menu`).fontSize(16).margin(5)
            .key('btn_Menu')
            .bindMenu(this.MyMenu3, {
              showInSubWindow: false
            })
        }
      }.width('100%')
    }.height('100%')
  }
}