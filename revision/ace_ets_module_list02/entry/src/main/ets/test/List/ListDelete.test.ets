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
import { describe, beforeEach, afterEach, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router'
import { Driver, Component, ON } from '@ohos.UiTest'
import CommonFunc from '../../MainAbility/common/Common';
import events_emitter from '@ohos.events.emitter';
import config from '@ohos.accessibility.config'
import accessibility from '@ohos.accessibility';
const CAPACITY: accessibility.Capability[] = ['retrieve', 'gesture'];
const BUNDLE_NAME = 'com.arkui.ace.scroll.list02/MainAbility';

export default function ListDeleteTest() {
  const SUITE: string = 'ListDelete'

  describe('ListDeleteTest', () => {
    beforeEach(async (done: Function) => {
      config.enableAbility(BUNDLE_NAME, CAPACITY).then(() => {
        console.info(`enable success`);
      }).catch((err: object) => {
        console.error(`failed to enable, ${err}`);
      })
      
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/ListDelete",
      }
      try {
        router.clear();
        let pages: router.RouterState = router.getState();
        if (pages == null || !("ListDelete" == pages.name)) {
          await router.pushUrl(options).then(() => {
            console.info(`${SUITE} router.pushUrl success`);
          })
        }
      } catch (err) {
        console.error(`${SUITE} beforeEach error:` + JSON.stringify(err));
      }
      done()
    });

    /*
     * @tc.number ArkUI_List_Delete_0100
     * @tc.name   ArkUI_List_Delete_0100
     * @tc.desc   Set swipeAction.end by CustomBuilder and edgeEffect is SwipeEdgeEffect.None
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Delete_0100', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Delete_0100'
      console.info(`${SUITE} ${CASE} START`);
      await sleep(2000);
      let baseKey: string = 'MyList1_ListItem0'
      let strJson: string = getInspectorByKey(baseKey);
      console.info("the rect json is " + strJson);

      expect(JSON.parse(strJson).$attrs.startDeleteAreaDistance).assertEqual("0.00vp");
      expect(JSON.parse(strJson).$attrs.endDeleteAreaDistance).assertEqual("0.00vp");

      let rectCenter: number[] = getRectCenter(JSON.parse(strJson).$rect)
      let driver: Driver = Driver.create()
      console.info("windowRect Left: " + globalThis.winLeft + ',' + "windowRect Top: " + globalThis.winTop);
      await driver.swipe(Math.round(rectCenter[0] + globalThis.winLeft), Math.round(rectCenter[1] + globalThis.winTop), 0, Math.round(rectCenter[1]) + globalThis.winTop)
      await sleep(3000);
      let strJsonButtonDel: string = getInspectorByKey(baseKey + '_Button_Del');

      expect(JSON.parse(strJsonButtonDel).$attrs.visibility).assertEqual("Visibility.Visible");
      let strJsonButtonSet: string = getInspectorByKey(baseKey + '_Button_Set');

      expect(JSON.parse(strJsonButtonSet).$attrs.visibility).assertEqual("Visibility.Visible");

      console.info(`${SUITE} ${CASE} END`);
      done();
    });

    /*
     * @tc.number ArkUI_List_Delete_0200
     * @tc.name   ArkUI_List_Delete_0200
     * @tc.desc   Set swipeAction.start by CustomBuilder and edgeEffect is SwipeEdgeEffect.None
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Delete_0200', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Delete_0200'
      console.info(`${SUITE} ${CASE} START`);

      let baseKey: string = 'MyList2_ListItem0'
      let strJson: string = getInspectorByKey(baseKey);
      expect(JSON.parse(strJson).$attrs.startDeleteAreaDistance).assertEqual("0.00vp");
      expect(JSON.parse(strJson).$attrs.endDeleteAreaDistance).assertEqual("0.00vp");

      let driver: Driver = Driver.create()
      console.info("windowRect Left: " + globalThis.winLeft + ',' + "windowRect Top: " + globalThis.winTop);
      let swipeRect = CommonFunc.getComponentRectOnScreen('MyList2_ListItem0');
      let top = swipeRect.top;
      let right = swipeRect.right;
      let left = swipeRect.left;
      let bottom = swipeRect.bottom;
      console.info("ArkUI_List_Delete_0200 value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom -top;
      let width = right - left;
      await driver.swipe(Math.round(left + width*0.4),Math.round(top + height/2),Math.round(left + width*0.9),Math.round(top + height/2));
      await sleep(3000);
      let strJsonButtonDel: string = getInspectorByKey(baseKey + '_Button_Del');
      expect(JSON.parse(strJsonButtonDel).$attrs.visibility).assertEqual("Visibility.Visible");
      let strJsonButtonSet: string = getInspectorByKey(baseKey + '_Button_Set');

      expect(JSON.parse(strJsonButtonSet).$attrs.visibility).assertEqual("Visibility.Visible");

      console.info(`${SUITE} ${CASE} END`);
      done();
    });

    /*
     * @tc.number ArkUI_List_Delete_0500
     * @tc.name   ArkUI_List_Delete_0500
     * @tc.desc   Set swipeAction.end by SwipeActionItem and edgeEffect is SwipeEdgeEffect.Spring
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Delete_0500', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Delete_0500'
      console.info(`${SUITE} ${CASE} START`);

      let driver: Driver = Driver.create()
      await driver.delayMs(1000)
      console.info("windowRect Left: " + globalThis.winLeft + ',' + "windowRect Top: " + globalThis.winTop);
      let baseKey: string = 'MyList5_ListItem0'
      let strJson: string = getInspectorByKey(baseKey);

      expect(JSON.parse(strJson).$attrs.startDeleteAreaDistance).assertEqual("0.00vp");
      expect(JSON.parse(strJson).$attrs.endDeleteAreaDistance).assertEqual("5.00vp");

      let rectCenter: number[] = getRectCenter(JSON.parse(strJson).$rect)
      await driver.swipe(Math.round(globalThis.winLeft + rectCenter[0] + (rectCenter[3] -rectCenter[2]) / 4), Math.round(globalThis.winTop + rectCenter[1]),
        0, Math.round(globalThis.winTop + rectCenter[1]))
      await sleep(3000);
      let strJson2: string = getInspectorByKey('MyList5ListItemOnDeleteH');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual("action trigger");
      console.info(`${SUITE} ${CASE} END`);
      done();
    });
  })
}

async function sleep(time: number) {
  return new Promise((resolve: (value: string) => void, reject) => {
    setTimeout(() => {
      resolve("sleep")
    }, time)
  }).then(() => {
    console.info(`sleep ${time} over...`)
  })
}

function getRectCenter(rect: string) {
  let regexp: RegExp = new RegExp('\\[', 'g');
  let rtn: string[] = rect.replace(regexp, '').replace(regexp, '').split(',')
  return [
    Number.parseFloat(rtn[0].trim()) + (Number.parseFloat(rtn[2].trim()) - Number.parseFloat(rtn[0].trim())) / 2,
    Number.parseFloat(rtn[1].trim()) + (Number.parseFloat(rtn[3].trim()) - Number.parseFloat(rtn[1].trim())) / 2,
    Number.parseFloat(rtn[0].trim()),
    Number.parseFloat(rtn[2].trim()),
  ]
}

function getRectCenterVp(rect: string) {
  let regexp: RegExp = new RegExp('\\[', 'g');
  let rtn = rect.replace(regexp, '').replace(regexp, '').split(',')
  return [
    px2vp(Number.parseFloat(rtn[0].trim()) + (Number.parseFloat(rtn[2].trim()) - Number.parseFloat(rtn[0].trim())) / 2),
    px2vp(Number.parseFloat(rtn[1].trim()) + (Number.parseFloat(rtn[3].trim()) - Number.parseFloat(rtn[1].trim())) / 2),
  ]
}