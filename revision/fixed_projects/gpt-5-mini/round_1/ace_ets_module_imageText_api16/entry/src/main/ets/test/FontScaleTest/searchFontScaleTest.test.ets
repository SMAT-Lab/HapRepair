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

export default function searchFontScaleTest() {
  describe('searchFontScaleTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/FontScaleTest/searchFontScaleTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get searchFontScaleTest state success " + JSON.stringify(pages));
        if (!("searchFontScaleTest" == pages.name)) {
          console.info("get searchFontScaleTest state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push searchFontScaleTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push searchFontScaleTest page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("searchFontScaleTest after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         searchFontScaleTest0001
     * @tc.desic        Test searchFontScaleTest abnormal value :undefined
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0001',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0001 START');
        await CommonFunc.sleep(1000);
        let search1 = getInspectorByKey('search1');
        let obj: ESObject = JSON.parse(search1);
        console.log('searchFontScaleTest0001---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0001---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0001] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0002
     * @tc.name         searchFontScaleTest0002
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.string')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0002',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0002 START');
        await CommonFunc.sleep(1000);
        let search2 = getInspectorByKey('search2');
        let obj: ESObject = JSON.parse(search2);
        console.log('searchFontScaleTest0002---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0002---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.500000");
        console.info("[searchFontScaleTest0002] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0003
     * @tc.name         searchFontScaleTest0003
     * @tc.desic        Test searchFontScaleTest Normal value 
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0003',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0003 START');
        await CommonFunc.sleep(1000);
        let search3 = getInspectorByKey('search3');
        let obj: ESObject = JSON.parse(search3);
        console.log('searchFontScaleTest0003---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0003---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.500000");
        console.info("[searchFontScaleTest0003] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0004
     * @tc.name         searchFontScaleTest0004
     * @tc.desic        Test searchFontScaleTest abnormal value :less than minimum
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0004',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0004 START');
        await CommonFunc.sleep(1000);
        let search4 = getInspectorByKey('search4');
        let obj: ESObject = JSON.parse(search4);
        console.log('searchFontScaleTest0004---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0004---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("1.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0004] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0005
     * @tc.name         searchFontScaleTest0005
     * @tc.desic        Test searchFontScaleTest abnormal value :Greater than maximum
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0005',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0005 START');
        await CommonFunc.sleep(1000);
        let search5 = getInspectorByKey('search5');
        let obj: ESObject = JSON.parse(search5);
        console.log('searchFontScaleTest0005---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0005---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("1.000000");
		expect(obj.$attrs.minFontScale).assertEqual("1.000000");
        console.info("[searchFontScaleTest0005] end");
    });

	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0006
     * @tc.name         searchFontScaleTest0006
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.boolean')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0006',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0006 START');
        await CommonFunc.sleep(1000);
        let search6 = getInspectorByKey('search6');
        let obj: ESObject = JSON.parse(search6);
        console.log('searchFontScaleTest0006---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0006---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0006] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0007
     * @tc.name         searchFontScaleTest0007
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.color')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0007',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0007 START');
        await CommonFunc.sleep(1000);
        let search7 = getInspectorByKey('search7');
        let obj: ESObject = JSON.parse(search7);
        console.log('searchFontScaleTest0007---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0007---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0007] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0008
     * @tc.name         searchFontScaleTest0008
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.float')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0008',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0008 START');
        await CommonFunc.sleep(1000);
        let search8 = getInspectorByKey('search8');
        let obj: ESObject = JSON.parse(search8);
        console.log('searchFontScaleTest0008---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0008---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.600000");
        console.info("[searchFontScaleTest0008] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0009
     * @tc.name         searchFontScaleTest0009
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.integer')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0009',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0009 START');
        await CommonFunc.sleep(1000);
        let search9 = getInspectorByKey('search9');
        let obj: ESObject = JSON.parse(search9);
        console.log('searchFontScaleTest0009---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0009---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("1.000000");
        console.info("[searchFontScaleTest0009] end");
    });
	
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0010
     * @tc.name         searchFontScaleTest00010
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.intarray')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0010',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0010 START');
        await CommonFunc.sleep(1000);
        let search10 = getInspectorByKey('search10');
        let obj: ESObject = JSON.parse(search10);
        console.log('searchFontScaleTest0010---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0010---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0010] end");
    });
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0011
     * @tc.name         searchFontScaleTest0011
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.pattern')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0011',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0011 START');
        await CommonFunc.sleep(1000);
        let search11 = getInspectorByKey('search11');
        let obj: ESObject = JSON.parse(search11);
        console.log('searchFontScaleTest0011---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0011---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0011] end");
    });
	
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0012
     * @tc.name         searchFontScaleTest0012
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.plural')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0012',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0012 START');
        await CommonFunc.sleep(1000);
        let search12 = getInspectorByKey('search12');
        let obj: ESObject = JSON.parse(search12);
        console.log('searchFontScaleTest0012---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0012---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0012] end");
    });
	
	
	/*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0013
     * @tc.name         searchFontScaleTest0013
     * @tc.desic        Test searchFontScaleTest reference value :$r('app.strarray')
	 * @tc.size         MediumTest
	 * @tc.type         Function
	 * @tc.level        1
     */
    it('searchFontScaleTest0013',  Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('searchFontScaleTest searchFontScaleTest0013 START');
        await CommonFunc.sleep(1000);
        let search13 = getInspectorByKey('search13');
        let obj: ESObject = JSON.parse(search13);
        console.log('searchFontScaleTest0013---' + obj.$attrs.maxFontScale)
		console.log('searchFontScaleTest0013---' + obj.$attrs.minFontScale)
        expect(obj.$attrs.maxFontScale).assertEqual("2.000000");
		expect(obj.$attrs.minFontScale).assertEqual("0.000000");
        console.info("[searchFontScaleTest0013] end");
    });
	

  })
}
