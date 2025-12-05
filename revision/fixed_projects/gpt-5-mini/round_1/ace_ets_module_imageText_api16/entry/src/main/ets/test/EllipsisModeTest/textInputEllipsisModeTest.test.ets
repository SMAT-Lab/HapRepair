/*
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

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium"
import { Driver, ON} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common'

export default function textInputEllipsisModeTest() {
  describe('textInputEllipsisModeTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/EllipsisModeTest/textInputEllipsisModeTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get textInputEllipsisModeTest state success " + JSON.stringify(pages));
        if (!("textInputEllipsisModeTest" == pages.name)) {
          console.info("get textInputEllipsisModeTest state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push textInputEllipsisModeTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push textInputEllipsisModeTest page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("textInputEllipsisModeTest after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         textInputEllipsisModeTest0001
     * @tc.desic        Test textInputEllipsisModeTest Normal value :EllipsisMode.END
	 * @tc.size         MediumTest
	 * @tc.type         Function
     */
    it('textInputEllipsisModeTest0001',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('textInputEllipsisModeTest textInputEllipsisModeTest0001 START');
        await CommonFunc.sleep(1000);
        let input1 = getInspectorByKey('input1');
        let obj: ESObject = JSON.parse(input1);
        console.log('textInputEllipsisModeTest0001---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.END");
        console.info("[textInputEllipsisModeTest0001] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0002
     * @tc.name         textInputEllipsisModeTest0002
     * @tc.desic        Test textInputEllipsisModeTest Normal value :EllipsisMode.START
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textInputEllipsisModeTest0002',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('textInputEllipsisModeTest textInputEllipsisModeTest0002 START');
        await CommonFunc.sleep(1000);
        let input2 = getInspectorByKey('input2');
        let obj: ESObject = JSON.parse(input2);
        console.log('textInputEllipsisModeTest0002---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.START");
        console.info("[textInputEllipsisModeTest0002] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0003
     * @tc.name         textInputEllipsisModeTest0003
     * @tc.desic        Test textInputEllipsisModeTest Normal value :EllipsisMode.CENTER
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textInputEllipsisModeTest0003',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('textInputEllipsisModeTest textInputEllipsisModeTest0003 START');
        await CommonFunc.sleep(1000);
        let input3 = getInspectorByKey('input3');
        let obj: ESObject = JSON.parse(input3);
        console.log('textInputEllipsisModeTest0003---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.CENTER");
        console.info("[textInputEllipsisModeTest0003] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0004
     * @tc.name         textInputEllipsisModeTest0004
     * @tc.desic        Test textInputEllipsisModeTest abnormal value :undefined
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textInputEllipsisModeTest0004',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('textInputEllipsisModeTest textInputEllipsisModeTest0004 START');
        await CommonFunc.sleep(1000);
        let input4 = getInspectorByKey('input4');
        let obj: ESObject = JSON.parse(input4);
        console.log('textInputEllipsisModeTest0004---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.END");
        console.info("[textInputEllipsisModeTest0004] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0005
     * @tc.name         textInputEllipsisModeTest0005
     * @tc.desic        Test textInputEllipsisModeTest abnormal value :null
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textInputEllipsisModeTest0005',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('textInputEllipsisModeTest textInputEllipsisModeTest0005 START');
        await CommonFunc.sleep(1000);
        let input5 = getInspectorByKey('input5');
        let obj: ESObject = JSON.parse(input5);
        console.log('textInputEllipsisModeTest0005---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.END");
        console.info("[textInputEllipsisModeTest0005] end");
    });
	
	
	
  })
}
