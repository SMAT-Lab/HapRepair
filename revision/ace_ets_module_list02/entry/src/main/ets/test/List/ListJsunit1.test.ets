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

export default function listJsunit1() {
  describe('list1Test', () => {
    beforeEach(async () => {
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/List1',
      }
      try {
        router.clear();
        await sleep(1);
        let pages = router.getState();
        console.info("get List state success " + JSON.stringify(pages));
        if (!("List1" == pages.name)) {
          console.info("get List state success " + JSON.stringify(pages.name));
          let result = await router.push(options)
          console.info("push List page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push List page error " + JSON.stringify(err));
      }
      await sleep(2)
    });

    afterEach(async () => {
      await sleep(1)
      console.info("List after each called");
    });

    /**
     * @tc.number listTest001
     * @tc.name listTest001
     * @tc.desc aceEtsTest
     * @tc.level Level 1
     * @tc.type
     * @tc.size
     */
    it('listTest001', Level.LEVEL1, async () => {
      console.info('++++++++++++++++ List TestCase 1 START ++++++++++++++++');
      await sleep(1)
      let strJson = getInspectorByKey('list01');
      let obj: Record<string, Object> = JSON.parse(strJson);
      console.info("get inspector value is: " + JSON.stringify(obj));
      console.log(JSON.stringify(obj.$type))
      expect(obj.$type).assertEqual('List')
      console.log('List‘s space is ' + JSON.stringify(JSON.parse(strJson).$attrs.space))
      console.log('List‘s initialIndex is ' + JSON.stringify(JSON.parse(strJson).$attrs.initialIndex))
      console.log('List‘s listDirection is ' + JSON.stringify(JSON.parse(strJson).$attrs.listDirection))
      console.log('List‘s divider is ' + JSON.stringify(JSON.parse(strJson).$attrs.divider))
      console.log('List‘s edgeEffect is ' + JSON.stringify(JSON.parse(strJson).$attrs.edgeEffect))
      expect(JSON.parse(strJson).$attrs.initialIndex).assertEqual('0')
      expect(JSON.parse(strJson).$attrs.listDirection).assertEqual('Axis.Vertical')
      expect(JSON.parse(strJson).$attrs.edgeEffect).assertEqual('EdgeEffect.None')
      expect(JSON.parse(strJson).$attrs.editMode).assertEqual(false)
      console.info('++++++++++++++++ List TestCase 1 End ++++++++++++++++');
    });
  })
}
