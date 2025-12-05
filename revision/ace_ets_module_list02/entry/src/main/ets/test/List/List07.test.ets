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
import hilog from '@ohos.hilog';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';

export default function List07Test() {
  describe('List07Test', () => {
    beforeEach(async (done: Function) => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/List07',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get list state success " + JSON.stringify(pages));
        if (!("List07" == pages.name)) {
          console.info("get list state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          console.info("push list page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push list page error: " + err);
      }
      done()
    });

    /**
     * @tc.number    ArkUX_Grid_List_Swiper_0250
     * @tc.name      ArkUX_Grid_List_Swiper_0250
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUX_Grid_List_Swiper_0250', Level.LEVEL1, async (done: Function) => {
      console.info('ArkUX_Grid_List_Swiper_0250 START');
      setTimeout(() => {
        try {
          let strJson = getInspectorByKey('ArkUX_Grid_List_Swiper_0250');
          console.info('ArkUX_Grid_List_Swiper_0250 START  :' + JSON.stringify(strJson));
          console.info("ArkUX_Grid_List_Swiper_0250 JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
          
          expect(JSON.parse(strJson).$attrs.laneGutter).assertEqual("20.00vp");
          console.info('ArkUX_Grid_List_Swiper_0250 END ');

        } catch (err) {
          expect().assertFail()
          console.info('ArkUX_Grid_List_Swiper_0250 ERR  ' + JSON.stringify(err));
        }
        done();
      }, 500)
    });

    /**
     * @tc.number    ArkUX_Grid_List_Swiper_0260
     * @tc.name      ArkUX_Grid_List_Swiper_0260
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUX_Grid_List_Swiper_0260', Level.LEVEL1, async (done: Function) => {
      console.info('ArkUX_Grid_List_Swiper_0260 START');
      setTimeout(() => {
        try {
          let strJson = getInspectorByKey('ArkUX_Grid_List_Swiper_0260');
          console.info('ArkUX_Grid_List_Swiper_0260 START  :' + JSON.stringify(strJson));
          console.info("ArkUX_Grid_List_Swiper_0260 JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
          expect(JSON.parse(strJson).$attrs.laneGutter).assertEqual("100.00vp");
          console.info('ArkUX_Grid_List_Swiper_0260 END ');
        } catch (err) {
          expect().assertFail()
          console.info('ArkUX_Grid_List_Swiper_0260 ERR  ' + JSON.stringify(err));
        }
        done();
      }, 500)
    });

    /**
     * @tc.number    ArkUX_Grid_List_Swiper_0270
     * @tc.name      ArkUX_Grid_List_Swiper_0270
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUX_Grid_List_Swiper_0270', Level.LEVEL1, async (done: Function) => {
      console.info('ArkUX_Grid_List_Swiper_0270 START');
      setTimeout(() => {
        try {
          let strJson = getInspectorByKey('ArkUX_Grid_List_Swiper_0270');
          console.info('ArkUX_Grid_List_Swiper_0270 START  :' + JSON.stringify(strJson));
          console.info("ArkUX_Grid_List_Swiper_0270 JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
          expect(JSON.parse(strJson).$attrs.laneGutter).assertEqual("0.00vp");
          console.info('ArkUX_Grid_List_Swiper_0270 END ');
        } catch (err) {
          expect().assertFail()
          console.info('ArkUX_Grid_List_Swiper_0270 ERR  ' + JSON.stringify(err));
        }
        done();
      }, 500)
    });

    /**
     * @tc.number    ArkUX_Grid_List_Swiper_0280
     * @tc.name      ArkUX_Grid_List_Swiper_0280
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUX_Grid_List_Swiper_0280', Level.LEVEL1, async (done: Function) => {
      console.info('ArkUX_Grid_List_Swiper_0280 START');
      setTimeout(() => {
        try {
          let strJson = getInspectorByKey('ArkUX_Grid_List_Swiper_0280');
          console.info('ArkUX_Grid_List_Swiper_0280 START  :' + JSON.stringify(strJson));
          console.info("ArkUX_Grid_List_Swiper_0280 JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
          expect(JSON.parse(strJson).$attrs.laneGutter).assertEqual("500.00vp");
          console.info('ArkUX_Grid_List_Swiper_0280 END ');
        } catch (err) {
          expect().assertFail()
          console.info('ArkUX_Grid_List_Swiper_0280 ERR  ' + JSON.stringify(err));
        }
        done();
      }, 500)
    });

    /**
     * @tc.number    ArkUX_Grid_List_Swiper_0290
     * @tc.name      ArkUX_Grid_List_Swiper_0290
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUX_Grid_List_Swiper_0290', Level.LEVEL1, async (done: Function) => {
      console.info('ArkUX_Grid_List_Swiper_0290 START');
      setTimeout(() => {
        try {
          let strJson = getInspectorByKey('ArkUX_Grid_List_Swiper_0290');
          console.info('ArkUX_Grid_List_Swiper_0290 START  :' + JSON.stringify(strJson));
          console.info("ArkUX_Grid_List_Swiper_0290 JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
          expect(JSON.parse(strJson).$attrs.laneGutter).assertEqual("250.00vp");
          console.info('ArkUX_Grid_List_Swiper_0290 END ');
        } catch (err) {
          expect().assertFail()
          console.info('ArkUX_Grid_List_Swiper_0290 ERR  ' + JSON.stringify(err));
        }
        done();
      }, 500)
    });
  })
}