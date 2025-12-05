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

export default function comSizeAbilityTest_16() {
  describe('comSizeAbilityTest_16', () => {
    beforeEach(async (done: Function) => {
      console.info("comSizeAbilityTest_16 beforeEach start")
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/comSizeAbility/comSizeAbilityTest_16'
      }
      try {
        router.clear()
        let pages = router.getState()
        console.info("get comSizeAbilityTest_16 state pages: " + JSON.stringify(pages))
        if (!("comSizeAbilityTest" == pages.name)) {
          console.info("get comSizeAbilityTest_16 state pages.name: " + JSON.stringify(pages.name))
          let result = await router.pushUrl(options)
          await CommonFunc.sleep(1000);
          console.info("push comSizeAbilityTest_16 success: " + JSON.stringify(result))
        }
      } catch (err) {
        console.error("push comSizeAbilityTest_16 page error: " + err)
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
    it('comSizeAbilityTestJsunit_41', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_41 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_16_to_1'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Tab_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Tab_Father");
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
      let button = await driver.findComponent(ON.id('Tab_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_41] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_41] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_41] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_41] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_41] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_41] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_41] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_41] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_41] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(240))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(140))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_41] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_42', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_42 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_16_to_2'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("TextArea_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("TextArea_Father");
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
      let button = await driver.findComponent(ON.id('TextArea_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_42] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_42] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_42] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_42] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_42] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_42] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_42] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_42] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_42] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
     
      expect(Math.floor(Math.abs(SizeWidth - vp2px(150))) <= 1).assertTrue();
      expect(Math.floor(Math.abs(SizeHeight - vp2px(120))) <= 1).assertTrue();

      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_42] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_43', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_43 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_16_to_3'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("TextClock_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("TextClock_Father");
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
      let button = await driver.findComponent(ON.id('TextClock_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_43] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_43] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_43] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_43] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_43] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_43] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_43] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_43] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_43] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(145))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(110))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_43] END')
      await CommonFunc.sleep(1000);
      done()
    })
  })
}