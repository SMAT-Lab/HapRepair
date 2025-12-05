/**
 * Copyright (c) 2022 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium'
import router from '@ohos.router';
import events_emitter from '@ohos.events.emitter'
import CommonFunc from '../MainAbility/common/Common';

export default function pageRouteTest() {
  describe('pageRouteTest', () => {
    beforeEach(async () => {
      console.info('pageRouteTest beforeEach start');
      let options: router.RouterOptions  = {
        url: 'MainAbility/pages/router/pageRoute',
      }
      // let result;
      try {
        router.clear();
        let pages = router.getState();
        console.info('get pageRouteTest pages: ' + JSON.stringify(pages));
        if (!('pageRoute' == pages.name)) {
          console.info('get pageRouteTest pages.name: ' + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info('pageRouteTest page result: ' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('pageRouteTest page error: ' + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('pageRouteTest after each called');
    })

    /**
     * @tc.number    testOnPageHide01
     * @tc.name      testOnPageHide01
     * @tc.desc      testOnPageHide01
     * @tc.level     Level 0
     * @tc.type
     * @tc.size
     */
    it('testOnPageHide01', Level.LEVEL0, async () => {
      console.info('[testOnPageHide01] START');
      let callback = (eventData: events_emitter.EventData) => {
        console.info('[testOnPageHide01] get event state result is: ' + JSON.stringify(eventData));
        expect(eventData.data?.value).assertEqual('onPageHide')

      }
      let innerEvent: events_emitter.InnerEvent = {
        eventId: 10,
        priority: events_emitter.EventPriority.LOW
      }
      try {
        events_emitter.on(innerEvent, callback)
        console.info('testOnPageHide01 click result is: ' + JSON.stringify(sendEventByKey('next', 10, '')));
      } catch (err) {
        console.info('[testOnPageHide01] on events_emitter err : ' + JSON.stringify(err));
      }
      console.info('[testOnPageHide01] testSendTouchEvent END');

    });
  })
}