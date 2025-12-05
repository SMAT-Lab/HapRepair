/**
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
import { describe, beforeEach, it, expect, Level } from "@ohos/hypium"
import router from '@ohos.router';
import CommonFunc from '../../MainAbility/common/Common';


export default function TextInputStopBackPressTest() {
  describe('TextInputStopBackPressTest', () => {
    beforeEach(async (done: Function) => {
      console.info("TextInputStopBackPressTest beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/TextInput/TextInputStopBackPress',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get TextInputStopBackPressTest state pages: " + JSON.stringify(pages));
        if (!("TextInputStopBackPress" == pages.name)) {
          console.info("get TextInputStopBackPressTest state pages.name: " + JSON.stringify(pages.name));
          let result = await router.replaceUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push TextInputStopBackPressTest success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push TextInputStopBackPressTest page error: " + err);
        expect().assertFail();
      }
      done();
    });
    
    /**
     * @tc.number    SUB_ACE_TextInputStopBackPressTest_0010
     * @tc.name      TextInputStopBackPressTest_0010
     * @tc.desc      test textInput StopBackPress: true
     * @tc.level:    Level 2
     * @tc.type : 
     * @tc.size :
     */
    it('TextInputStopBackPressTest_0010', Level.LEVEL2, async (done: Function) => {
      console.info('[TextInputStopBackPressTest_0010] START');
      let strJson = getInspectorByKey('TextInput_stopBackPress_true');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[TextInputStopBackPressTest_0010] component obj is: " + JSON.stringify(obj.$attrs));
      expect(obj.$attrs.stopBackPress).assertEqual(true);      
      console.info('[TextInputStopBackPressTest_0010] END');
      done();
    });

    /**
     * @tc.number    SUB_ACE_TextInputStopBackPressTest_0020
     * @tc.name      TextInputStopBackPressTest_0020
     * @tc.desc      test textInput StopBackPress: false
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('TextInputStopBackPressTest_0020', Level.LEVEL2, async (done: Function) => {
      console.info('[TextInputStopBackPressTest_0020] START');
      let strJson = getInspectorByKey('TextInput_stopBackPress_false');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[TextInputStopBackPressTest_0020] component obj is: " + JSON.stringify(obj.$attrs));
      expect(obj.$attrs.stopBackPress).assertEqual(false);
      console.info('[TextInputStopBackPressTest_0020] END');
      done();
    });

    /**
     * @tc.number    SUB_ACE_TextInputStopBackPressTest_0030
     * @tc.name      TextInputStopBackPressTest_0030
     * @tc.desc      test textInput StopBackPress: default
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('TextInputStopBackPressTest_0030', Level.LEVEL2, async (done: Function) => {
      console.info('[TextInputStopBackPressTest_0030] START');
      let strJson = getInspectorByKey('TextInput_stopBackPress_default');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[TextInputStopBackPressTest_0030] component obj is: " + JSON.stringify(obj.$attrs));
      expect(obj.$attrs.stopBackPress).assertEqual(true);
      console.info('[TextInputStopBackPressTest_0030] END');
      done();
    });

    /**
     * @tc.number    SUB_ACE_TextInputStopBackPressTest_0040
     * @tc.name      TextInputStopBackPressTest_0040
     * @tc.desc      test textInput StopBackPress: undefined
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('TextInputStopBackPressTest_0040', Level.LEVEL2, async (done: Function) => {
      console.info('[TextInputStopBackPressTest_0040] START');
      let strJson = getInspectorByKey('TextInput_stopBackPress_undefined');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[TextInputStopBackPressTest_0040] component obj is: " + JSON.stringify(obj.$attrs));
      expect(obj.$attrs.stopBackPress).assertEqual(true);
      console.info('[TextInputStopBackPressTest_0040] END');
      done();
    });
  })
}

