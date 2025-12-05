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

import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium"
import router from '@ohos.router';
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';

export default function addOrDelText() {
  describe('addOrDelText', () => {
    beforeEach(async (done: Function) => {
      console.info("addOrDelText beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/TextArea/addOrDelText',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get addOrDelText state pages: " + JSON.stringify(pages));
        if (!("addOrDelText" == pages.name)) {
          console.info("get addOrDelText state pages.name: " + JSON.stringify(pages.name));
          let result = await router.replaceUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push addOrDelText success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push addOrDelText page error: " + err);
        expect().assertFail();
      }
      done();
    });
    

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         addOrDelText0001
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0001', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0001 START');
        await CommonFunc.sleep(500);
        let add = await driver.findComponent(ON.id('add'));
        await CommonFunc.sleep(500);
        await add.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0001---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("文本123");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0001---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0001---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("文本123");
        
        console.info("[addOrDelText0001] end");
        done();
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0002
     * @tc.name         addOrDelText0002
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0002', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0002 START');
        await CommonFunc.sleep(500);
        let del11 = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del11.click();
        await CommonFunc.sleep(500);
        let add11 = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add11.click();
        await CommonFunc.sleep(500);
        let add = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0002---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("文本添加文本");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0002---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0002---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("文本添加文本");
        
        console.info("[addOrDelText0002] end");
        done();

    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0003
     * @tc.name         addOrDelText0003
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0003', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0003 START');
        await CommonFunc.sleep(500);
        let del11 = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del11.click();
        await CommonFunc.sleep(500);
        let add11 = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add11.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0003---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0003---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0003---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("");
        
        console.info("[addOrDelText0003] end");
        done();

    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0004
     * @tc.name         addOrDelText0004
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0004', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0004 START');
        await CommonFunc.sleep(500);
        let del11 = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del11.click();
        await CommonFunc.sleep(500);
        let add11 = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add11.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del3'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0004---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("添文本");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0004---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0004---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("");
        
        console.info("[addOrDelText0004] end");
        done();

    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0005
     * @tc.name         addOrDelText0005
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0005', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0005 START');
        await CommonFunc.sleep(500);
        let del11 = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del11.click();
        await CommonFunc.sleep(500);
        let add11 = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add11.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del4'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0005---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0005---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0005---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("");
        
        console.info("[addOrDelText0005] end");
        done();

    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0006
     * @tc.name         addOrDelText0006
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0006', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0006 START');
        await CommonFunc.sleep(500);
        let del11 = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del11.click();
        await CommonFunc.sleep(500);
        let add11 = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add11.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del5'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0006---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0006---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("加文本");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0006---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("加文本");
        
        console.info("[addOrDelText0006] end");
        done();

    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0007
     * @tc.name         addOrDelText0007
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0007', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0007 START');
        await CommonFunc.sleep(500);
        let del11 = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del11.click();
        await CommonFunc.sleep(500);
        let add11 = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add11.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del6'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0007---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0007---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("添加");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0007---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("添加");
        
        console.info("[addOrDelText0007] end");
        done();

    });


    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0008
     * @tc.name         addOrDelText0008
     * @tc.desic        The addText and deleteText methods are added to the 
                          textinput, search, and textarea components.
     */
    it('addOrDelText0008', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('addOrDelText addOrDelText0008 START');
        await CommonFunc.sleep(500);
        let del11 = await driver.findComponent(ON.id('del2'));
        await CommonFunc.sleep(500);
        await del11.click();
        await CommonFunc.sleep(500);
        let add11 = await driver.findComponent(ON.id('add2'));
        await CommonFunc.sleep(500);
        await add11.click();
        await CommonFunc.sleep(500);
        let del = await driver.findComponent(ON.id('del7'));
        await CommonFunc.sleep(500);
        await del.click();
        await CommonFunc.sleep(500);

        let strJson = getInspectorByKey('textArea');
        let obj: ESObject = JSON.parse(strJson);
        console.log('addOrDelText0008---obj.$attrs.text---' +
         JSON.stringify(obj.$attrs.text));
        expect('' + obj.$attrs.text).assertEqual("");

        let strJson2 = getInspectorByKey('textInput');
        let obj2: ESObject = JSON.parse(strJson2);
        console.log('addOrDelText0008---obj2.$attrs.text---' +
         JSON.stringify(obj2.$attrs.text));
        expect('' + obj2.$attrs.text).assertEqual("");

        let strJson3 = getInspectorByKey('search');
        let obj3: ESObject = JSON.parse(strJson3);
        console.log('addOrDelText0008---obj3.$attrs.value---' +
         JSON.stringify(obj3.$attrs.value));
        expect('' + obj3.$attrs.value).assertEqual("");
        
        console.info("[addOrDelText0008] end");
        done();

    });


  })
}