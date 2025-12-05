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

export default function comSizeAbilityTest_13() {
  describe('comSizeAbilityTest_13', () => {
    beforeEach(async (done: Function) => {
      console.info("comSizeAbilityTest_13 beforeEach start")
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/comSizeAbility/comSizeAbilityTest_13'
      }
      try {
        router.clear()
        let pages = router.getState()
        console.info("get comSizeAbilityTest_13 state pages: " + JSON.stringify(pages))
        if (!("comSizeAbilityTest" == pages.name)) {
          console.info("get comSizeAbilityTest_13 state pages.name: " + JSON.stringify(pages.name))
          let result = await router.pushUrl(options)
          await CommonFunc.sleep(1000);
          console.info("push comSizeAbilityTest_13 success: " + JSON.stringify(result))
        }
      } catch (err) {
        console.error("push comSizeAbilityTest_13 page error: " + err)
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
    it('comSizeAbilityTestJsunit_33', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_33 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_13_to_1'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("RichText_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("RichText_Father");
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
      let button = await driver.findComponent(ON.id('RichText_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_33] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_33] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_33] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_33] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_33] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_33] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_33] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_33] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_33] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(210))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(132))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_33] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_34', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_34 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_13_to_2'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Scroll_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Scroll_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let Father_Size = modePosition1.size
      let Father_SizeWidth = Father_Size.width
      let Father_SizeHeight = Father_Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Scroll_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_34] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_34] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_34] component Father_Size is: " + JSON.stringify(Father_Size))
      console.info("[comSizeAbilityTestJsunit_34] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_34] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_34] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_34] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_34] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_34] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_34] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.floor(Math.abs(SizeWidth - Father_SizeWidth * 0.7)) <= 1).assertTrue();
      expect(Math.floor(Math.abs(SizeHeight - Father_SizeHeight * 0.7)) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_34] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_35', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_35 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_13_to_3'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Search_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Search_Father");
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
      let button = await driver.findComponent(ON.id('Search_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_35] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_35] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_35] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_35] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_35] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_35] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_35] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_35] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_35] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(360))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(240))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_35] END')
      await CommonFunc.sleep(1000);
      done()
    })
  })
}