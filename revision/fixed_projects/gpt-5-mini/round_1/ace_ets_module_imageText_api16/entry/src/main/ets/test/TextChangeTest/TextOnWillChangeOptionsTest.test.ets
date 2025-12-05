/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

export default function TextOnWillChangeOptionsTest() {
  describe('TextOnWillChangeOptionsTest', () => {
    beforeEach(async (done: Function) => {
      console.info("TextOnWillChangeOptionsTest beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/TextChangeTest/TextOnWillChangeOptionsTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get TextOnWillChangeOptionsTest state success " + JSON.stringify(pages));
        let result = await router.replaceUrl(options);
        if (!("TextOnWillChangeOptionsTest" == pages.name)) {
          console.info("get TextOnWillChangeOptionsTest state success " + JSON.stringify(pages.name));
          await CommonFunc.sleep(2000);
          console.info("push TextOnWillChangeOptionsTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push TextOnWillChangeOptionsTest page error: " + err);
      }
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("TextOnWillChangeOptionsTest after each called");
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0001
    * @tc.name         TextOnWillChangeOptionsTest0001
    * @tc.desic        Test search Add text
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('TextOnWillChangeOptionsTest0001', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('TextOnWillChangeOptionsTest TextOnWillChangeOptionsTest0001 START');
        await CommonFunc.sleep(1000);

        let search = await driver.findComponent(ON.id('search'));
        await CommonFunc.sleep(500);
        let point = await search.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.inputText(point,'3')
        await CommonFunc.sleep(500);

        let oldContent = getInspectorByKey('oldContent');
        let oldContentText:string = JSON.parse(oldContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + oldContentText)

        let newContent = getInspectorByKey('newContent');
        let newContentText:string = JSON.parse(newContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + newContentText)

        let oldPreviewTextValue = getInspectorByKey('oldPreviewTextValue');
        let oldPreviewTextValueText:string = JSON.parse(oldPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + oldPreviewTextValueText)

        let oldPreviewTextOffset = getInspectorByKey('oldPreviewTextOffset');
        let oldPreviewTextOffsetText:string = JSON.parse(oldPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + oldPreviewTextOffsetText)

        let newPreviewTextValue = getInspectorByKey('newPreviewTextValue');
        let newPreviewTextValueText:string = JSON.parse(newPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + newPreviewTextValueText)

        let newPreviewTextOffset = getInspectorByKey('newPreviewTextOffset');
        let newPreviewTextOffsetText:string = JSON.parse(newPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + newPreviewTextOffsetText)

        let rangeBefore = getInspectorByKey('rangeBefore');
        let rangeBeforeText:string = JSON.parse(rangeBefore).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + rangeBeforeText)

        let rangeAfter = getInspectorByKey('rangeAfter');
        let rangeAfterText:string = JSON.parse(rangeAfter).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0001---' + rangeAfterText)

        expect(oldContentText).assertEqual('12');
        expect(newContentText).assertEqual('123');
        expect(oldPreviewTextValueText).assertEqual('');
        expect(oldPreviewTextOffsetText).assertEqual('-1');
        expect(newPreviewTextValueText).assertEqual('');
        expect(newPreviewTextOffsetText).assertEqual('-1');
        expect(rangeBeforeText).assertEqual('2-2');
        expect(rangeAfterText).assertEqual('2-3');
        console.info("[TextOnWillChangeOptionsTest0001] end");
        done();
    });


    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0002
    * @tc.name         TextOnWillChangeOptionsTest0002
    * @tc.desic        Test search delete text
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('TextOnWillChangeOptionsTest0002', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('TextOnWillChangeOptionsTest TextOnWillChangeOptionsTest0002 START');

        await CommonFunc.sleep(1000);
        let search = await driver.findComponent(ON.id('search'));
        await CommonFunc.sleep(500);
        search.click();
        await CommonFunc.sleep(500);
        await driver.triggerKey(2055);
        await CommonFunc.sleep(500);

        let oldContent = getInspectorByKey('oldContent');
        let oldContentText:string = JSON.parse(oldContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + oldContentText)

        let newContent = getInspectorByKey('newContent');
        let newContentText:string = JSON.parse(newContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + newContentText)

        let oldPreviewTextValue = getInspectorByKey('oldPreviewTextValue');
        let oldPreviewTextValueText:string = JSON.parse(oldPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + oldPreviewTextValueText)

        let oldPreviewTextOffset = getInspectorByKey('oldPreviewTextOffset');
        let oldPreviewTextOffsetText:string = JSON.parse(oldPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + oldPreviewTextOffsetText)

        let newPreviewTextValue = getInspectorByKey('newPreviewTextValue');
        let newPreviewTextValueText:string = JSON.parse(newPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + newPreviewTextValueText)

        let newPreviewTextOffset = getInspectorByKey('newPreviewTextOffset');
        let newPreviewTextOffsetText:string = JSON.parse(newPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + newPreviewTextOffsetText)

        let rangeBefore = getInspectorByKey('rangeBefore');
        let rangeBeforeText:string = JSON.parse(rangeBefore).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + rangeBeforeText)

        let rangeAfter = getInspectorByKey('rangeAfter');
        let rangeAfterText:string = JSON.parse(rangeAfter).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0002---' + rangeAfterText)

        expect(oldContentText).assertEqual('12');
        expect(newContentText).assertEqual('1');
        expect(oldPreviewTextValueText).assertEqual('');
        expect(oldPreviewTextOffsetText).assertEqual('-1');
        expect(newPreviewTextValueText).assertEqual('');
        expect(newPreviewTextOffsetText).assertEqual('-1');
        expect(rangeBeforeText).assertEqual('1-2');
        expect(rangeAfterText).assertEqual('1-1');
        console.info("[TextOnWillChangeOptionsTest0002] end");
        done();
    });


    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0003
    * @tc.name         TextOnWillChangeOptionsTest0003
    * @tc.desic        Test TextArea Add text
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('TextOnWillChangeOptionsTest0003', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('TextOnWillChangeOptionsTest TextOnWillChangeOptionsTest0003 START');
        await CommonFunc.sleep(1000);

        let textArea = await driver.findComponent(ON.id('textArea'));
        await CommonFunc.sleep(500);
        let point = await textArea.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.inputText(point,'3')
        await CommonFunc.sleep(500);

        let oldContent = getInspectorByKey('oldContent');
        let oldContentText:string = JSON.parse(oldContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + oldContentText)

        let newContent = getInspectorByKey('newContent');
        let newContentText:string = JSON.parse(newContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + newContentText)

        let oldPreviewTextValue = getInspectorByKey('oldPreviewTextValue');
        let oldPreviewTextValueText:string = JSON.parse(oldPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + oldPreviewTextValueText)

        let oldPreviewTextOffset = getInspectorByKey('oldPreviewTextOffset');
        let oldPreviewTextOffsetText:string = JSON.parse(oldPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + oldPreviewTextOffsetText)

        let newPreviewTextValue = getInspectorByKey('newPreviewTextValue');
        let newPreviewTextValueText:string = JSON.parse(newPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + newPreviewTextValueText)

        let newPreviewTextOffset = getInspectorByKey('newPreviewTextOffset');
        let newPreviewTextOffsetText:string = JSON.parse(newPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + newPreviewTextOffsetText)

        let rangeBefore = getInspectorByKey('rangeBefore');
        let rangeBeforeText:string = JSON.parse(rangeBefore).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + rangeBeforeText)

        let rangeAfter = getInspectorByKey('rangeAfter');
        let rangeAfterText:string = JSON.parse(rangeAfter).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0003---' + rangeAfterText)

        expect(oldContentText).assertEqual('12');
        expect(newContentText).assertEqual('123');
        expect(oldPreviewTextValueText).assertEqual('');
        expect(oldPreviewTextOffsetText).assertEqual('-1');
        expect(newPreviewTextValueText).assertEqual('');
        expect(newPreviewTextOffsetText).assertEqual('-1');
        expect(rangeBeforeText).assertEqual('2-2');
        expect(rangeAfterText).assertEqual('2-3');
        console.info("[TextOnWillChangeOptionsTest0003] end");
        done();
    });


    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0004
    * @tc.name         TextOnWillChangeOptionsTest0004
    * @tc.desic        Test TextArea delete text
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('TextOnWillChangeOptionsTest0004', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('TextOnWillChangeOptionsTest TextOnWillChangeOptionsTest0004 START');

        await CommonFunc.sleep(1000);
        let textArea = await driver.findComponent(ON.id('textArea'));
        await CommonFunc.sleep(500);
        textArea.click();
        await CommonFunc.sleep(500);
        await driver.triggerKey(2055);
        await CommonFunc.sleep(500);

        let oldContent = getInspectorByKey('oldContent');
        let oldContentText:string = JSON.parse(oldContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + oldContentText)

        let newContent = getInspectorByKey('newContent');
        let newContentText:string = JSON.parse(newContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + newContentText)

        let oldPreviewTextValue = getInspectorByKey('oldPreviewTextValue');
        let oldPreviewTextValueText:string = JSON.parse(oldPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + oldPreviewTextValueText)

        let oldPreviewTextOffset = getInspectorByKey('oldPreviewTextOffset');
        let oldPreviewTextOffsetText:string = JSON.parse(oldPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + oldPreviewTextOffsetText)

        let newPreviewTextValue = getInspectorByKey('newPreviewTextValue');
        let newPreviewTextValueText:string = JSON.parse(newPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + newPreviewTextValueText)

        let newPreviewTextOffset = getInspectorByKey('newPreviewTextOffset');
        let newPreviewTextOffsetText:string = JSON.parse(newPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + newPreviewTextOffsetText)

        let rangeBefore = getInspectorByKey('rangeBefore');
        let rangeBeforeText:string = JSON.parse(rangeBefore).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + rangeBeforeText)

        let rangeAfter = getInspectorByKey('rangeAfter');
        let rangeAfterText:string = JSON.parse(rangeAfter).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0004---' + rangeAfterText)

        expect(oldContentText).assertEqual('12');
        expect(newContentText).assertEqual('1');
        expect(oldPreviewTextValueText).assertEqual('');
        expect(oldPreviewTextOffsetText).assertEqual('-1');
        expect(newPreviewTextValueText).assertEqual('');
        expect(newPreviewTextOffsetText).assertEqual('-1');
        expect(rangeBeforeText).assertEqual('1-2');
        expect(rangeAfterText).assertEqual('1-1');
        console.info("[TextOnWillChangeOptionsTest0004] end");
        done();
    });


    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0005
    * @tc.name         TextOnWillChangeOptionsTest0005
    * @tc.desic        Test TextInput Add text
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('TextOnWillChangeOptionsTest0005', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('TextOnWillChangeOptionsTest TextOnWillChangeOptionsTest0005 START');
        await CommonFunc.sleep(1000);

        let textInput = await driver.findComponent(ON.id('textInput'));
        await CommonFunc.sleep(500);
        let point = await textInput.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.inputText(point,'3')
        await CommonFunc.sleep(500);

        let oldContent = getInspectorByKey('oldContent');
        let oldContentText:string = JSON.parse(oldContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + oldContentText)

        let newContent = getInspectorByKey('newContent');
        let newContentText:string = JSON.parse(newContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + newContentText)

        let oldPreviewTextValue = getInspectorByKey('oldPreviewTextValue');
        let oldPreviewTextValueText:string = JSON.parse(oldPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + oldPreviewTextValueText)

        let oldPreviewTextOffset = getInspectorByKey('oldPreviewTextOffset');
        let oldPreviewTextOffsetText:string = JSON.parse(oldPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + oldPreviewTextOffsetText)

        let newPreviewTextValue = getInspectorByKey('newPreviewTextValue');
        let newPreviewTextValueText:string = JSON.parse(newPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + newPreviewTextValueText)

        let newPreviewTextOffset = getInspectorByKey('newPreviewTextOffset');
        let newPreviewTextOffsetText:string = JSON.parse(newPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + newPreviewTextOffsetText)

        let rangeBefore = getInspectorByKey('rangeBefore');
        let rangeBeforeText:string = JSON.parse(rangeBefore).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + rangeBeforeText)

        let rangeAfter = getInspectorByKey('rangeAfter');
        let rangeAfterText:string = JSON.parse(rangeAfter).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0005---' + rangeAfterText)

        expect(oldContentText).assertEqual('12');
        expect(newContentText).assertEqual('123');
        expect(oldPreviewTextValueText).assertEqual('');
        expect(oldPreviewTextOffsetText).assertEqual('-1');
        expect(newPreviewTextValueText).assertEqual('');
        expect(newPreviewTextOffsetText).assertEqual('-1');
        expect(rangeBeforeText).assertEqual('2-2');
        expect(rangeAfterText).assertEqual('2-3');
        console.info("[TextOnWillChangeOptionsTest0005] end");
        done();
    });


    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0006
    * @tc.name         TextOnWillChangeOptionsTest0006
    * @tc.desic        Test TextInput delete text
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('TextOnWillChangeOptionsTest0006', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('TextOnWillChangeOptionsTest TextOnWillChangeOptionsTest0006 START');

        await CommonFunc.sleep(1000);
        let textInput = await driver.findComponent(ON.id('textInput'));
        await CommonFunc.sleep(500);
        textInput.click();
        await CommonFunc.sleep(500);
        await driver.triggerKey(2055);
        await CommonFunc.sleep(500);

        let oldContent = getInspectorByKey('oldContent');
        let oldContentText:string = JSON.parse(oldContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + oldContentText)

        let newContent = getInspectorByKey('newContent');
        let newContentText:string = JSON.parse(newContent).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + newContentText)

        let oldPreviewTextValue = getInspectorByKey('oldPreviewTextValue');
        let oldPreviewTextValueText:string = JSON.parse(oldPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + oldPreviewTextValueText)

        let oldPreviewTextOffset = getInspectorByKey('oldPreviewTextOffset');
        let oldPreviewTextOffsetText:string = JSON.parse(oldPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + oldPreviewTextOffsetText)

        let newPreviewTextValue = getInspectorByKey('newPreviewTextValue');
        let newPreviewTextValueText:string = JSON.parse(newPreviewTextValue).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + newPreviewTextValueText)

        let newPreviewTextOffset = getInspectorByKey('newPreviewTextOffset');
        let newPreviewTextOffsetText:string = JSON.parse(newPreviewTextOffset).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + newPreviewTextOffsetText)

        let rangeBefore = getInspectorByKey('rangeBefore');
        let rangeBeforeText:string = JSON.parse(rangeBefore).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + rangeBeforeText)

        let rangeAfter = getInspectorByKey('rangeAfter');
        let rangeAfterText:string = JSON.parse(rangeAfter).$attrs.content;
        console.log('TextOnWillChangeOptionsTest0006---' + rangeAfterText)

        expect(oldContentText).assertEqual('12');
        expect(newContentText).assertEqual('1');
        expect(oldPreviewTextValueText).assertEqual('');
        expect(oldPreviewTextOffsetText).assertEqual('-1');
        expect(newPreviewTextValueText).assertEqual('');
        expect(newPreviewTextOffsetText).assertEqual('-1');
        expect(rangeBeforeText).assertEqual('1-2');
        expect(rangeAfterText).assertEqual('1-1');
        console.info("[TextOnWillChangeOptionsTest0006] end");
        done();
    });
  })
}