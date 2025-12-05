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
import { expect, it, afterEach, afterAll, beforeEach, beforeAll, describe, Level } from '@ohos/hypium';
import router from '@ohos.router';
import {
  PointerMatrix,
  WindowMode,
  ResizeDirection,
  DisplayRotation,
  MatchPattern,
  BY,
  ON,
  UiWindow,
  Driver,
  Component,
  UiComponent,
  UiDriver
} from '@ohos.UiTest'
import CommonFunc from '../../MainAbility/common/Common';
import componentUtils from '@ohos.arkui.componentUtils';

export default function comSizeAbilityTest_07() {
  describe('comSizeAbilityTest_07', () => {
    beforeEach(async (done: Function) => {
      console.info("comSizeAbilityTest_07 beforeEach start")
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/comSizeAbility/comSizeAbilityTest_07'
      }
      try {
        router.clear()
        let pages = router.getState()
        console.info("get comSizeAbilityTest_07 state pages: " + JSON.stringify(pages))
        if (!("comSizeAbilityTest" == pages.name)) {
          console.info("get comSizeAbilityTest_07 state pages.name: " + JSON.stringify(pages.name))
          let result = await router.pushUrl(options)
          await CommonFunc.sleep(1000);
          console.info("push comSizeAbilityTest_07 success: " + JSON.stringify(result))
        }
      } catch (err) {
        console.error("push comSizeAbilityTest_07 page error: " + err)
        expect().assertFail()
      }
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_20', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_20 START')
      await CommonFunc.sleep(1000);
      if (globalThis.winLeft == 0) {
        let driver = await Driver.create();
        let button1 = await driver.findComponent(ON.id('btn_MenuItemGroup'))
        await button1.click()
        await CommonFunc.sleep(2000)
        let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("MenuItemGroup_Size");
        let Translate = modePosition.translate
        let Transform = modePosition.transform
        let Scale = modePosition.scale
        let Rotate = modePosition.rotate
        console.info("[comSizeAbilityTestJsunit_20] component Translate is: " + JSON.stringify(Translate))
        console.info("[comSizeAbilityTestJsunit_20] component Transform is: " + JSON.stringify(Transform))
        console.info("[comSizeAbilityTestJsunit_20] component Scale is: " + JSON.stringify(Scale))
        console.info("[comSizeAbilityTestJsunit_20] component Rotate is: " + JSON.stringify(Rotate))
        await CommonFunc.sleep(1000);
        expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
        expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
        expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
        expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      }
      console.info('[comSizeAbilityTestJsunit_20] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_21', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_21 START')
      await CommonFunc.sleep(1000);
      if (globalThis.winLeft == 0) {
        let driver = await Driver.create();
        let button1 = await driver.findComponent(ON.id('btn_MenuItem'))
        await button1.click()
        await CommonFunc.sleep(2000);
        let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("MenuItem_Size");
        let Size = modePosition.size
        let SizeWidth = Size.width
        let SizeHeight = Size.height
        let Translate = modePosition.translate
        let Transform = modePosition.transform
        let Scale = modePosition.scale
        let Rotate = modePosition.rotate
        console.info("[comSizeAbilityTestJsunit_21] component Size is: " + JSON.stringify(Size))
        console.info("[comSizeAbilityTestJsunit_21] component Translate is: " + JSON.stringify(Translate))
        console.info("[comSizeAbilityTestJsunit_21] component Transform is: " + JSON.stringify(Transform))
        console.info("[comSizeAbilityTestJsunit_21] component Scale is: " + JSON.stringify(Scale))
        console.info("[comSizeAbilityTestJsunit_21] component Rotate is: " + JSON.stringify(Rotate))
        await CommonFunc.sleep(1000);
        expect(Math.abs(SizeWidth - Math.round(vp2px(90))) <= 1).assertTrue();
        expect(Math.abs(SizeHeight - Math.round(vp2px(90))) <= 1).assertTrue();
        expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
        expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
        expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
        expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      }
      console.info('[comSizeAbilityTestJsunit_21] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_22', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_22 START')
      await CommonFunc.sleep(1000);
      if (globalThis.winLeft == 0) {
        let driver = await Driver.create()
        let button1 = await driver.findComponent(ON.id('btn_Menu'))
        await button1.click()
        await CommonFunc.sleep(2000);
        let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Menu_Size");
        let Size = modePosition.size
        let SizeWidth = Size.width
        let SizeHeight = Size.height
        let Translate = modePosition.translate
        let Transform = modePosition.transform
        let Scale = modePosition.scale
        let Rotate = modePosition.rotate
        console.info("[comSizeAbilityTestJsunit_22] component Size is: " + JSON.stringify(Size))
        console.info("[comSizeAbilityTestJsunit_22] component Translate is: " + JSON.stringify(Translate))
        console.info("[comSizeAbilityTestJsunit_22] component Transform is: " + JSON.stringify(Transform))
        console.info("[comSizeAbilityTestJsunit_22] component Scale is: " + JSON.stringify(Scale))
        console.info("[comSizeAbilityTestJsunit_22] component Rotate is: " + JSON.stringify(Rotate))
        await CommonFunc.sleep(1000);
        expect(Math.abs(SizeWidth - Math.round(vp2px(100))) <= 1).assertTrue();
        expect(Math.abs(SizeHeight - Math.round(vp2px(100))) <= 1).assertTrue();
        expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
        expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
        expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
        expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
        await CommonFunc.sleep(1000);
        await driver.click(Math.round(globalThis.winLeft + vp2px(200)), Math.round(globalThis.winTop + vp2px(200)));
        await CommonFunc.sleep(1000);
      }
      console.info('[comSizeAbilityTestJsunit_22] END')
      await CommonFunc.sleep(1000);
      done()
    })
  })
}