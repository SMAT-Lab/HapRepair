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

export default function searchButton() {
  describe('searchButton', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Search/searchButton',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get searchButton state success " + JSON.stringify(pages));
        if (!("searchButton" == pages.name)) {
          console.info("get searchButton state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push searchButton page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push searchButton page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("searchButton after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         searchButton0001
     * @tc.desic        When the content of the search component is 
                          empty, the search button is dimmed
     */
    it('searchButton0001', Level.LEVEL1, async () => {
        let driver = Driver.create();
        console.info('searchButton searchButton0001 START');
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('search1');
        let obj: ESObject = JSON.parse(strJson);
        console.log('searchButton0001---obj.$attrs.searchButtonOption.autoDisable---' +
         JSON.stringify(obj.$attrs.searchButtonOption.autoDisable));
        expect('' + obj.$attrs.searchButtonOption.autoDisable).assertEqual("true");
        //true
        let strJson2 = getInspectorByKey('search2');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('searchButton0001---obj2.$attrs.searchButtonOption.autoDisable---' +
         JSON.stringify(obj.$attrs.searchButtonOption.autoDisable));
        expect('' + obj2.$attrs.searchButtonOption.autoDisable).assertEqual("false");
        //false
        let strJson3 = getInspectorByKey('search3');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('searchButton0001---obj3.$attrs.searchButtonOption.autoDisable---' +
         JSON.stringify(obj.$attrs.searchButtonOption.autoDisable));
        expect('' + obj3.$attrs.searchButtonOption.autoDisable).assertEqual("false");
        //false
        console.info("[searchButton0001] end");
    });
  })
}
