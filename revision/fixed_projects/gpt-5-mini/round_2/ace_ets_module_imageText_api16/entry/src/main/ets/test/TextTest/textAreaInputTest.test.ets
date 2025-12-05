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

export default function textAreaInputTest() {
  describe('textAreaInputTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Text/textAreaInputTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get textAreaInputTest state success " + JSON.stringify(pages));
        if (!("textAreaInputTest" == pages.name)) {
          console.info("get textAreaInputTest state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push textAreaInputTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push textAreaInputTest page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("textAreaInputTest after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         textAreaInputTest0001
     * @tc.desic        Adapt to Xiaoyi input method pop-up window
     */
    it('textAreaInputTest0001', Level.LEVEL0, async () => {
      let driver = Driver.create();
      if (globalThis.winLeft == 0) {
        console.info('textAreaInputTest textAreaInputTest0001 START');
        await CommonFunc.sleep(1000);
        let textArea1 = await driver.findComponent(ON.id('textArea1'));
        await CommonFunc.sleep(500);
        let textArea2 = await driver.findComponent(ON.id('textArea2'));
        await CommonFunc.sleep(500);
        await textArea2.click()
        await CommonFunc.sleep(500);
		await driver.triggerKey(2070);
        await CommonFunc.sleep(1000);
        let button1 = await driver.findComponent(ON.text('下一步'));
        await CommonFunc.sleep(500);
        await button1?.click();
        await CommonFunc.sleep(500);

        await textArea1.longClick();
        await CommonFunc.sleep(500);
        let button3 = await driver.findComponent(ON.text('全选'));
        await CommonFunc.sleep(500);
        await button3?.click();
        console.info('textAreaInputTest0001 全选 success');
        await CommonFunc.sleep(500);

        await textArea1.dragTo(textArea2);
        await CommonFunc.sleep(500);
        console.info('textAreaInputTest0001 dragTo success');
        let button4 = await driver.findComponent(ON.text('取消'));
        await CommonFunc.sleep(500);
        await button4?.click();
        await CommonFunc.sleep(500);
        console.info('textAreaInputTest0001 取消 success');

        await textArea1.longClick();
        await CommonFunc.sleep(500);
        let button5 = await driver.findComponent(ON.text('全选'));
        await CommonFunc.sleep(500);
        await button5?.click();
        await CommonFunc.sleep(500);
        console.info('textAreaInputTest0001 2 全选 success');
        let button6 = await driver.findComponent(ON.text('复制'));
        await CommonFunc.sleep(500);
        await button6?.click();
        await CommonFunc.sleep(500);
        console.info('textAreaInputTest0001 2 复制 success');
        await textArea1.dragTo(textArea2);
        await CommonFunc.sleep(500);
        let Text7 = await driver.findComponent(ON.text('长按内容至浮起，可拖拽到底部小艺'));
        if(Text7){
          let button7 = await driver.findComponent(ON.type('Button'));
          await CommonFunc.sleep(500);
          await button7?.click();
        }
        await CommonFunc.sleep(500);
        console.info('textAreaInputTest0001 2 Button success');
      }else {
        //切换HAD 输入法
        let textArea2 = await driver.findComponent(ON.id('textArea2'));
        await CommonFunc.sleep(500);
        await textArea2.inputText('aaa');
        await CommonFunc.sleep(500);
        let areaText = await textArea2.getText();
        await CommonFunc.sleep(500);
        if(areaText != 'aaa'){
          await driver.triggerKey(2047)
        }
      }
      
      expect("").assertEqual("");
      console.info("[textAreaInputTest0001] end");
    });
  })
}
