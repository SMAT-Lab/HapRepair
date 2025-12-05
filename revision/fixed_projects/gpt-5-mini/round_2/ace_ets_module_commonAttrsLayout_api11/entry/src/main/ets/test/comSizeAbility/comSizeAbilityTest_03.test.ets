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
  UiComponent,
  UiDriver,
  Component,
  Driver,
  UiWindow,
  ON,
  BY,
  MatchPattern,
  DisplayRotation,
  ResizeDirection,
  WindowMode,
  PointerMatrix
} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';
import componentUtils from '@ohos.arkui.componentUtils';

export default function comSizeAbilityTest_03() {
  describe('comSizeAbilityTest_03', () => {
    beforeEach(async (done: Function) => {
      console.info("comSizeAbilityTest_03 beforeEach start")
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/comSizeAbility/comSizeAbilityTest_03'
      }
      try {
        router.clear()
        let pages = router.getState()
        console.info("get comSizeAbilityTest_03 state pages: " + JSON.stringify(pages))
        if (!("comSizeAbilityTest" == pages.name)) {
          console.info("get comSizeAbilityTest_03 state pages.name: " + JSON.stringify(pages.name))
          let result = await router.pushUrl(options)
          await CommonFunc.sleep(1000);
          console.info("push comSizeAbilityTest_03 success: " + JSON.stringify(result))
        }
      } catch (err) {
        console.error("push comSizeAbilityTest_03 page error: " + err)
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
    it('comSizeAbilityTestJsunit_07', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_07 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_03_to_1'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Circle_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Circle_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Circle_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_07] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_07] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_07] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_07] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_07] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_07] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_07] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_07] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_07] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(310))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(140))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_07] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_08', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_08 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_03_to_2'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Column_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Column_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Column_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_08] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_08] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_08] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_08] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_08] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_08] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_08] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_08] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_08] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(160))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(100))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_08] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_09', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_09 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_03_to_3'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Counter_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Counter_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Counter_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_09] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_09] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_09] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_09] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_09] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_09] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_09] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_09] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_09] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(150))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(100))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_09] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_10', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_10 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_03_to_4'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Divider_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Divider_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Divider_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_10] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_10] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_10] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_10] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_10] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_10] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_10] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_10] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_10] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(130))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(100))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_10] END')
      await CommonFunc.sleep(1000);
      done()
    })
  })
}