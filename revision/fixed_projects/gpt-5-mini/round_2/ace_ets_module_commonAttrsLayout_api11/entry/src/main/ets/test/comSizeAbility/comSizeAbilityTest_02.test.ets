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

export default function comSizeAbilityTest_02() {
  describe('comSizeAbilityTest_02', () => {
    beforeEach(async (done: Function) => {
      console.info("comSizeAbilityTest_02 beforeEach start")
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/comSizeAbility/comSizeAbilityTest_02'
      }
      try {
        router.clear()
        let pages = router.getState()
        console.info("get comSizeAbilityTest_02 state pages: " + JSON.stringify(pages))
        if (!("comSizeAbilityTest" == pages.name)) {
          console.info("get comSizeAbilityTest_02 state pages.name: " + JSON.stringify(pages.name))
          let result = await router.pushUrl(options)
          await CommonFunc.sleep(1000);
          console.info("push comSizeAbilityTest_02 success: " + JSON.stringify(result))
        }
      } catch (err) {
        console.error("push comSizeAbilityTest_02 page error: " + err)
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
    it('comSizeAbilityTestJsunit_05', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_05 START')
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("CheckboxGroup_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("CheckboxGroup_Father");
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
      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('CheckboxGroup_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_05] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_05] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_05] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_05] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_05] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_05] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_05] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_05] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_05] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(90))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(60))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_05] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_06', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_06 START')
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Checkbox_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Checkbox_Father");
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
      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('Checkbox_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_06] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_06] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_06] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_06] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_06] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_06] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_06] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_06] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_06] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(90))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(60))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_06] END')
      await CommonFunc.sleep(1000);
      done()
    })
  })
}