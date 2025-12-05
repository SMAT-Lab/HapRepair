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

export default function textAreaEllipsisModeTest() {
  describe('textAreaEllipsisModeTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/EllipsisModeTest/textAreaEllipsisModeTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get textAreaEllipsisModeTest state success " + JSON.stringify(pages));
        if (!("textAreaEllipsisModeTest" == pages.name)) {
          console.info("get textAreaEllipsisModeTest state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push textAreaEllipsisModeTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push textAreaEllipsisModeTest page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("textAreaEllipsisModeTest after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         textAreaEllipsisModeTest0001
     * @tc.desic        Test textAreaEllipsisModeTest Normal value :EllipsisMode.END
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textAreaEllipsisModeTest0001', Level.LEVEL1, async () => {
        let driver = Driver.create();
        console.info('textAreaEllipsisModeTest textAreaEllipsisModeTest0001 START');
        await CommonFunc.sleep(1000);
        let area1 = getInspectorByKey('area1');
        let obj: ESObject = JSON.parse(area1);
        console.log('textAreaEllipsisModeTest0001---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.END");
        console.info("[textAreaEllipsisModeTest0001] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0002
     * @tc.name         textAreaEllipsisModeTest0002
     * @tc.desic        Test textAreaEllipsisModeTest Normal value :EllipsisMode.START
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textAreaEllipsisModeTest0002', Level.LEVEL1, async () => {
        let driver = Driver.create();
        console.info('textAreaEllipsisModeTest textAreaEllipsisModeTest0002 START');
        await CommonFunc.sleep(1000);
        let area2 = getInspectorByKey('area2');
        let obj: ESObject = JSON.parse(area2);
        console.log('textAreaEllipsisModeTest0002---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.START");
        console.info("[textAreaEllipsisModeTest0002] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0003
     * @tc.name         textAreaEllipsisModeTest0003
     * @tc.desic        Test textAreaEllipsisModeTest Normal value :EllipsisMode.CENTER
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textAreaEllipsisModeTest0003', Level.LEVEL1, async () => {
        let driver = Driver.create();
        console.info('textAreaEllipsisModeTest textAreaEllipsisModeTest0003 START');
        await CommonFunc.sleep(1000);
        let area3 = getInspectorByKey('area3');
        let obj: ESObject = JSON.parse(area3);
        console.log('textAreaEllipsisModeTest0003---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.CENTER");
        console.info("[textAreaEllipsisModeTest0003] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0004
     * @tc.name         textAreaEllipsisModeTest0004
     * @tc.desic        Test textAreaEllipsisModeTest abnormal value :undefined
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textAreaEllipsisModeTest0004', Level.LEVEL1, async () => {
        let driver = Driver.create();
        console.info('textAreaEllipsisModeTest textAreaEllipsisModeTest0004 START');
        await CommonFunc.sleep(1000);
        let area4 = getInspectorByKey('area4');
        let obj: ESObject = JSON.parse(area4);
        console.log('textAreaEllipsisModeTest0004---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.END");
        console.info("[textAreaEllipsisModeTest0004] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0005
     * @tc.name         textAreaEllipsisModeTest0005
     * @tc.desic        Test textAreaEllipsisModeTest abnormal value :null
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('textAreaEllipsisModeTest0005', Level.LEVEL1, async () => {
        let driver = Driver.create();
        console.info('textAreaEllipsisModeTest textAreaEllipsisModeTest0005 START');
        await CommonFunc.sleep(1000);
        let area5 = getInspectorByKey('area5');
        let obj: ESObject = JSON.parse(area5);
        console.log('textAreaEllipsisModeTest0005---' + obj.$attrs.ellipsisMode)
        expect(obj.$attrs.ellipsisMode).assertEqual("EllipsisMode.END");
        console.info("[textAreaEllipsisModeTest0005] end");
    });
	
	
	
  })
}
