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

export default function searchHalfLeadingTest() {
  describe('searchHalfLeadingTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/HalfLeadingTest/searchHalfLeadingTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get searchHalfLeadingTest state success " + JSON.stringify(pages));
        if (!("searchHalfLeadingTest" == pages.name)) {
          console.info("get searchHalfLeadingTest state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push searchHalfLeadingTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push searchHalfLeadingTest page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("searchHalfLeadingTest after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         searchHalfLeadingTest0001
     * @tc.desic        Test searchHalfLeadingTest Normal value :true
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchHalfLeadingTest0001',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchHalfLeadingTest searchHalfLeadingTest0001 START');
        await CommonFunc.sleep(1000);
        let search1 = getInspectorByKey('search1');
        let obj: ESObject = JSON.parse(search1);
        console.log('searchHalfLeadingTest0001---' + obj.$attrs.halfLeading)
        expect(obj.$attrs.halfLeading).assertEqual(true);
        console.info("[searchHalfLeadingTest0001] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0002
     * @tc.name         searchHalfLeadingTest0002
     * @tc.desic        Test searchHalfLeadingTest Normal value :false
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchHalfLeadingTest0002',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchHalfLeadingTest searchHalfLeadingTest0002 START');
        await CommonFunc.sleep(1000);
        let search2 = getInspectorByKey('search2');
        let obj: ESObject = JSON.parse(search2);
        console.log('searchHalfLeadingTest0002---' + obj.$attrs.halfLeading)
        expect(obj.$attrs.halfLeading).assertEqual(false);
        console.info("[searchHalfLeadingTest0002] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0003
     * @tc.name         searchHalfLeadingTest0003
     * @tc.desic        Test searchHalfLeadingTest abnormal value :undefined
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchHalfLeadingTest0003',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchHalfLeadingTest searchHalfLeadingTest0003 START');
        await CommonFunc.sleep(1000);
        let search3 = getInspectorByKey('search3');
        let obj: ESObject = JSON.parse(search3);
        console.log('searchHalfLeadingTest0003---' + obj.$attrs.halfLeading)
        expect(obj.$attrs.halfLeading).assertEqual(false);
        console.info("[searchHalfLeadingTest0003] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0004
     * @tc.name         searchHalfLeadingTest0004
     * @tc.desic        Test searchHalfLeadingTest abnormal value :null
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchHalfLeadingTest0004',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchHalfLeadingTest searchHalfLeadingTest0004 START');
        await CommonFunc.sleep(1000);
        let search4 = getInspectorByKey('search4');
        let obj: ESObject = JSON.parse(search4);
        console.log('searchHalfLeadingTest0004---' + obj.$attrs.halfLeading)
        expect(obj.$attrs.halfLeading).assertEqual(false);
        console.info("[searchHalfLeadingTest0004] end");
    });
	
  })
}
