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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function NavigationAdd() {
  describe('NavigationAdd',  () =>{
    beforeEach(async  (done:Function) =>{
      console.info("NavigationAdd beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/navigationAdd',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavigationAdd state success " + JSON.stringify(pages));
        if (pages.name !== 'NavigationAdd') { 
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavigationAdd page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavigationAdd page error: " + err);
      }
      done()
    });

    afterEach(async  () =>{
      await CommonFunc.sleep(1000);
      console.info("NavigationAdd after each called");
    });

    /*
     * @tc.number      apiCommponentAddJsunit_1000
     * @tc.name        apiCommponentAddJsunit_1000
     * @tc.desc        apiCommponentAddJsunit_1000
     * @tc.level       Level 1
     */
    it('apiCommponentAddJsunit_1000', Level.LEVEL1, async (done: Function) => {
      console.info('apiCommponentAddJsunit_1000 START');
      router.clear();
      router.pushUrl({
        url: 'MainAbility/pages/Navigation/navigationAdd',
        params: {
          data1: 'message',
          data2: {
            data3: [123, 456, 789]
          }
        }
      }).then(() => {
        // success
      }).catch((err: BusinessError) => {
        console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
      })
      let strJson = getInspectorByKey('navigationAdd');
      let obj: ESObject = JSON.parse(strJson);
      console.info('[apiCommponentAddJsunit_1000] component obj is: ' + obj.$type);
      expect(obj.$type).assertEqual('Navigation');
      done();
    });

    /*
     * @tc.number      apiCommponentAddJsunit_2300
     * @tc.name        apiCommponentAddJsunit_2300
     * @tc.desc        apiCommponentAddJsunit_2300
     * @tc.level       Level 1
     */
    it('apiCommponentAddJsunit_2300', Level.LEVEL1, async (done: Function) => {
      console.info("[apiCommponentAddJsunit_2300] START" );
      router.clear();
      router.pushUrl({
        url: 'MainAbility/pages/Navigation/navition3',
        params: {
          data1: 'message',
          data2: {
            data3: [123, 456, 789]
          }
        }
      }).then(() => {
        // success
      }).catch((err: BusinessError) => {
        console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
      })
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('ArkUX_Navigation_1111');
      console.info('apiCommponentAddJsunit_2300 START  :'+ JSON.stringify(strJson));
      let obj: ESObject = JSON.parse(strJson);
      console.info("apiCommponentAddJsunit_2300 obj is: " + JSON.stringify(obj));
      let mode: string = obj.$attrs.mode
      let minContentWidth: string = obj.$attrs.minContentWidth
      let navBarWidth: string = obj.$attrs.navBarWidth
      let navBarWidthRange: string = obj.$attrs.navBarWidthRange
      expect(mode).assertEqual('NavigationMode.AUTO');
      expect(minContentWidth).assertEqual('50.00vp');
      expect(navBarWidth).assertEqual('100.00vp');
      expect(navBarWidthRange).assertEqual('50.00vp, 300.00vp');
      console.info('apiCommponentAddJsunit_2300 END ');
      done();
    });

    /*
     * @tc.number      apiCommponentAddJsunit_1500
     * @tc.name        apiCommponentAddJsunit_1500
     * @tc.desc        apiCommponentAddJsunit_1500
     * @tc.level       Level 1
     */
    it('apiCommponentAddJsunit_1500', Level.LEVEL1, async (done: Function) => {
       console.info("[apiCommponentAddJsunit_1500] START" );
       router.clear();
       router.pushUrl({
        url: 'MainAbility/pages/Navigation/navrouterAdd',
        params: {
          data1: 'message',
          data2: {
            data3: [123, 456, 789]
          }
        }
      }).then(() => {
        // success
      }).catch((err: BusinessError) => {
        console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
      })
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('addOne');
      let obj: ESObject = JSON.parse(strJson);
      console.info('[apiCommponentAddJsunit_1500] component obj is: ' + obj.$type);
      expect(obj.$type).assertEqual('Navigation');
      done();   
    });

  })
}