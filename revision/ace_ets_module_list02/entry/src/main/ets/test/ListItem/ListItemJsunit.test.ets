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

import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';

function sleep(time: number) {
  return new Promise((resolve: (value: string) => void) => {
    setTimeout(() => {
      resolve("ok")
    }, time * 1000)
  }).then(() => {
    console.info(`sleep ${time} over...`)
  })
}

export default function listItemJsunit() {


  describe('listItemTest', () => {
    beforeEach(async () => {
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/ListItem/ListItem',
      }
      try {
        router.clear();
        await sleep(1);
        let pages = router.getState();
        console.info("get ListItem state success " + JSON.stringify(pages));
        if (!("ListItem" == pages.name)) {
          console.info("get ListItem state success " + JSON.stringify(pages.name));
          let result = await router.push(options)
          console.info("push ListItem page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push ListItem page error " + JSON.stringify(err));
      }
      await sleep(2)
    });

    afterEach(async () => {
      await sleep(1)
      console.info("ListItem after each called");
    });

    /**
     * @tc.number listItemTest001
     * @tc.name listItemTest001
     * @tc.desc aceEtsTest
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('listItemTest001', Level.LEVEL1, async () => {
      console.info('++++++++++++++++ ListItem TestCase 1 START ++++++++++++++++');
      await sleep(1)
      let strJson = getInspectorByKey('listitem');
      console.log(JSON.stringify(strJson))
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('ListItem')
      console.log(JSON.stringify(JSON.parse(strJson).$attrs.type))
      console.log('ListItem1‘s sticky is ' + JSON.stringify(JSON.parse(strJson).$attrs.sticky))
      console.log('ListItem1‘s editable is ' + JSON.stringify(JSON.parse(strJson).$attrs.editable))
      expect(JSON.parse(strJson).$attrs.sticky).assertEqual('Sticky.None')
      console.info('++++++++++++++++ ListItem TestCase 1 End ++++++++++++++++');
    });

    /**
     * @tc.number listItemTest002
     * @tc.name listItemTest002
     * @tc.desc aceEtsTest
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('listItemTest002', Level.LEVEL1, async () => {
      console.info('++++++++++++++++ ListItem TestCase 2 START ++++++++++++++++');
      let strJson = getInspectorByKey('listitem2');
      let obj: Record<string, Object> = JSON.parse(strJson);
      console.info("get inspector value is: " + JSON.stringify(obj));
      console.log(JSON.stringify(obj.$type))
      expect(obj.$type).assertEqual('ListItem')
      console.log(JSON.stringify(JSON.parse(strJson).$attrs.type))
      console.log('ListItem2‘s sticky is ' + JSON.stringify(JSON.parse(strJson).$attrs.sticky))
      console.log('ListItem2‘s editable is ' + JSON.stringify(JSON.parse(strJson).$attrs.editable))
      await sleep(1)
      expect(JSON.parse(strJson).$attrs.sticky).assertEqual('Sticky.Normal')
      console.info('++++++++++++++++ ListItem TestCase 2 End ++++++++++++++++');
    });

    /**
     * @tc.number listItemTest003
     * @tc.name listItemTest003
     * @tc.desc aceEtsTest
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('listItemTest003', Level.LEVEL1, async () => {
      console.info('++++++++++++++++ ListItem TestCase 3 START ++++++++++++++++');
      let strJson = getInspectorByKey('listitem3');
      let obj: Record<string, Object> = JSON.parse(strJson);
      console.info("get inspector value is: " + JSON.stringify(obj));
      console.log(JSON.stringify(obj.$type))
      expect(obj.$type).assertEqual('ListItem')
      console.log(JSON.stringify(JSON.parse(strJson).$attrs.type))
      console.log('ListItem3‘s sticky is ' + JSON.stringify(JSON.parse(strJson).$attrs.sticky))
      console.log('ListItem3‘s editable is ' + JSON.stringify(JSON.parse(strJson).$attrs.editable))
      await sleep(1)
      expect(JSON.parse(strJson).$attrs.sticky).assertEqual('Sticky.Opacity')
      console.info('++++++++++++++++ ListItem TestCase 3 End ++++++++++++++++');
    });

    /**
     * @tc.number listItemTest004
     * @tc.name listItemTest004
     * @tc.desc aceEtsTest
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('listItemTest004', Level.LEVEL1, async () => {
      console.info('++++++++++++++++ ListItem TestCase 4 START ++++++++++++++++');
      let strJson = getInspectorByKey('listitem4');
      let obj: Record<string, Object> = JSON.parse(strJson);
      console.info("get inspector value is: " + JSON.stringify(obj));
      console.log(JSON.stringify(obj.$type))
      expect(obj.$type).assertEqual('ListItem')
      console.log(JSON.stringify(JSON.parse(strJson).$attrs.type))
      console.log('ListItem4‘s sticky is ' + JSON.stringify(JSON.parse(strJson).$attrs.sticky))
      console.log('ListItem4‘s editable is ' + JSON.stringify(JSON.parse(strJson).$attrs.editable))
      await sleep(1)
      expect(JSON.parse(strJson).$attrs.sticky).assertEqual('Sticky.Opacity')
      console.info('++++++++++++++++ ListItem TestCase 4 End ++++++++++++++++');
    });
  })
}
