/**
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/utils/Common';
const waitUiReadyMs = 1000;

export default function RouterWithParamJsunit() {
  describe('RouterWithParamJsunit', () =>{
    beforeEach(async  (done:Function) =>{
      console.info("RouterWithParamJsunit beforeEach start");
      let options:router.RouterOptions = {
        url: 'MainAbility/pages/State/RouterPushThree',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get RouterWithParamJsunit state pages: " + JSON.stringify(pages));
        if (!("RouterPushThree" == pages.name)) {
          console.info("get RouterWithParamJsunit state pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          console.info("push RouterWithParamJsunit page result:" + JSON.stringify(result));
          router.clear()
        }
      } catch (err) {
        console.error("push RouterWithParamJsunit page error:" + err);
      }
      done()
    });

    afterEach(async () =>{
      await CommonFunc.sleep(1000);
      router.clear()
    });

    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_021
     * @tc.name      RouterWithParamJsunit_001
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterWithParamJsunit_001', Level.LEVEL2, async  (done:Function) =>{
      console.info('[RouterWithParamJsunit_001] START');
      let driver = Driver.create();
      await driver.delayMs(waitUiReadyMs);
      let btn = await driver.findComponent(ON.id('turn_RouterPushUrlOne_null'));
      await driver.delayMs(waitUiReadyMs);
      await btn.click();
      await driver.delayMs(waitUiReadyMs);
      let strJson = getInspectorByKey('Text_RouterPushThree_Params');
      
      console.info("[RouterWithParamJsunit_001] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual("RouterPushThree undefined");
      console.info('[RouterWithParamJsunit_001] END');
      done();
    });

    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_020
     * @tc.name      RouterWithParamJsunit_002
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterWithParamJsunit_002', Level.LEVEL2, async  (done:Function) =>{
      console.info('[RouterWithParamJsunit_002] START');
      let driver = Driver.create();
      await driver.delayMs(waitUiReadyMs);
      let btn = await driver.findComponent(ON.id('turn_RouterPushUrlOne_string_number'));
      await driver.delayMs(waitUiReadyMs);
      await btn.click();
      await driver.delayMs(waitUiReadyMs);
      let strJson = getInspectorByKey('Text_RouterPushThree_Params');
      
      console.info("[RouterWithParamJsunit_002] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('RouterPushThree {"data1":{"data2":"message","data3":12345}}')
      console.info('[RouterWithParamJsunit_002] END');
      done();
    });

    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_022
     * @tc.name      RouterWithParamJsunit_003
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterWithParamJsunit_003', Level.LEVEL2, async  (done:Function) =>{
      console.info('[RouterWithParamJsunit_003] START');
      let driver = Driver.create();
      await driver.delayMs(waitUiReadyMs);      
      let btn1 = await driver.findComponent(ON.id('turn_RouterPushThree_stringArray'));
      await driver.delayMs(waitUiReadyMs);
      await btn1.click();
      await driver.delayMs(waitUiReadyMs);
      let strJson = getInspectorByKey('Text_RouterPushThree_Params');
      
      console.info("[RouterWithParamJsunit_003] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('RouterPushThree {"data1":{"data2":["message","message1"]}}')
      console.info('[RouterWithParamJsunit_003] END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_018
     * @tc.name      RouterWithParamJsunit_004
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterWithParamJsunit_004', Level.LEVEL2, async  (done:Function) =>{
      console.info('[RouterWithParamJsunit_004] START');
      let driver = Driver.create();
      await driver.delayMs(waitUiReadyMs);
      let btn = await driver.findComponent(ON.id('turn_RouterPushUrlOne_string_number1'));
      await driver.delayMs(waitUiReadyMs);
      await btn.click();
      await driver.delayMs(waitUiReadyMs);
      let strJson = getInspectorByKey('Text_RouterPushThree_Params');
      
      console.info("[RouterWithParamJsunit_004] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('RouterPushThree {"data1":"message","data2":12345}')
      console.info('[RouterWithParamJsunit_004] END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_019
     * @tc.name      RouterWithParamJsunit_005
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterWithParamJsunit_005', Level.LEVEL2, async  (done:Function) =>{
      console.info('[RouterWithParamJsunit_005] START');
      let driver = Driver.create();
      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let btn = await scrollBar.scrollSearch(ON.id('turn_RouterPushUrlOne_string_numArray'));
      await driver.delayMs(waitUiReadyMs);
      await btn.click();
      await driver.delayMs(waitUiReadyMs);
      let strJson = getInspectorByKey('Text_RouterPushThree_Params');
      
      console.info("[RouterWithParamJsunit_005] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('RouterPushThree {"data1":"message","data2":[1,2,3,4,5]}')
      console.info('[RouterWithParamJsunit_005] END');
      done();
    });

    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_015
     * @tc.name      RouterWithParamJsunit_006
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterWithParamJsunit_006', Level.LEVEL2, async  (done:Function) =>{
      console.info('[RouterWithParamJsunit_006] START');
      let driver = Driver.create();
      await driver.delayMs(waitUiReadyMs);
      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let btn = await scrollBar.scrollSearch(ON.id('turn_RouterPushUrlOne_string_number_push'));
      await driver.delayMs(waitUiReadyMs);
      await btn.click();
      await driver.delayMs(waitUiReadyMs);
      let strJson = getInspectorByKey('Text_RouterPushThree_Params');
      
      console.info("[RouterWithParamJsunit_006] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('RouterPushThree {"data1":{"data2":"message","data3":12345}}')
      console.info('[RouterWithParamJsunit_006] END');
      done();
    });
  
    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_017
     * @tc.name      RouterWithParamJsunit_007
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterWithParamJsunit_007', Level.LEVEL2, async  (done:Function) =>{
      console.info('[RouterWithParamJsunit_007] START');
      let driver = Driver.create();
      await driver.delayMs(waitUiReadyMs);
      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let btn = await scrollBar.scrollSearch(ON.id('turn_RouterPushUrlOne_null_push'));
      await driver.delayMs(waitUiReadyMs);
      await btn.click();
      await driver.delayMs(waitUiReadyMs);
      let strJson = getInspectorByKey('Text_RouterPushThree_Params');
      
      console.info("[RouterWithParamJsunit_007] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('RouterPushThree undefined')
      console.info('[RouterWithParamJsunit_007] END');
      done();
    });
  })
}
