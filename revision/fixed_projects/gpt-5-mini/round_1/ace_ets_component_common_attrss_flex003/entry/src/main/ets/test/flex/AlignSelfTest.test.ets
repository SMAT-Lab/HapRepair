/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

import Utils from '../../MainAbility/common/Utils';
import CommonTest from '../common/CommonTest';
import { describe, expect } from '@ohos/hypium'

class PageConfig {
  testName: string = '';
  pageName: string = '';
  pageUrl: string = '';
  parentWidth: number = 0;
  parentHeight: number = 0;
  parentComponentKey: string = '';  
}

class TestValue {
  describe: string = '';
  setValue: SetValue = new SetValue();
  expectValue: ESObject;
}

class AlignItems {
  Row: VerticalAlign = VerticalAlign.Center;
  Column: HorizontalAlign = HorizontalAlign.Center;
  Flex: ItemAlign = ItemAlign.Auto;
  GridRow: ItemAlign = ItemAlign.Auto;
}

class SetValue {
  defaultAlignItems: AlignItems = new AlignItems();
  alignSelf: ItemAlign = ItemAlign.Auto;
}

export default function alignSelfTest() {

  let supportView = [
    'Navigator',
    'TextTimer',
    'Progress',
    'QRCode',
    'Radio',
    'Rating',
    'Search',
    'Slider',
    'Text'
  ]

  let containerView = ['Row', 'Column', 'FlexRow', 'FlexRowReverse', 'FlexColumn', 'FlexColumnReverse']

  let parentWidth = 400;
  let parentHeight = 400;

  let pageConfig: PageConfig = {
    testName: 'testAlignSelf',
    pageName: 'AlignSelfPage',
    pageUrl: 'MainAbility/pages/flex/AlignSelfPage',
    parentWidth: parentWidth,
    parentHeight: parentHeight,
    parentComponentKey: 'parentComponentKey'
  }

  let defaultAlignItems: AlignItems = {
    Row: VerticalAlign.Top,
    Column: HorizontalAlign.Start,
    Flex: ItemAlign.Start,
    GridRow: ItemAlign.Start
  }

  let testValues: TestValue[] = [{
    describe: 'SetAuto',
    setValue: {
      defaultAlignItems: defaultAlignItems,
      alignSelf: ItemAlign.Auto
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Auto'
    }
  }, {
    describe: 'SetStart',
    setValue: {
      defaultAlignItems: {
        Row: VerticalAlign.Center,
        Column: HorizontalAlign.Center,
        Flex: ItemAlign.Center,
        GridRow: ItemAlign.Center
      },
      alignSelf: ItemAlign.Start
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Start'
    }
  }, {
    describe: 'SetCenter',
    setValue: {
      defaultAlignItems: defaultAlignItems,
      alignSelf: ItemAlign.Center
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Center'
    }
  }, {
    describe: 'SetEnd',
    setValue: {
      defaultAlignItems: defaultAlignItems,
      alignSelf: ItemAlign.End
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.End'
    }
  }, {
    describe: 'SetStretch',
    setValue: {
      defaultAlignItems: defaultAlignItems,
      alignSelf: ItemAlign.Stretch
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetBaseLine',
    setValue: {
      defaultAlignItems: defaultAlignItems,
      alignSelf: ItemAlign.Baseline
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Baseline'
    }
  }]

  describe('AlignSelfTest', () => {
    CommonTest.initTestByContainerView(pageConfig, containerView, supportView, testValues, async (data: ESObject) => {

      //Get component rect and parent component rect
      let parentRect = Utils.getComponentRect(data.pageConfig.parentComponentKey);
      let rect = Utils.getComponentRect(data.pageConfig.componentKey);

      //Get viewObj in order to judge attribute is set success
      let viewObj: ESObject = Utils.getComponentByKey(data.pageConfig.componentKey);

      //Verify attribute
      expect(viewObj.$attrs.alignSelf).assertEqual(data.testValue.expectValue.expectAlignSelf);

      //When set different value, judge the alignment is correct
      if (data.pageConfig.containerView == 'Row' || data.pageConfig.containerView == 'FlexRow') {
        //When alignSelf : ItemAlign.Auto or ItemAlign.Start
        if (data.testValue.describe == 'SetAuto' || data.testValue.describe == 'SetStart') {
          expect(Math.floor(Math.abs(rect.top - parentRect.top)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.left - parentRect.left)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetCenter') {
          //When alignSelf : ItemAlign.Center
          let rectY = Utils.getRectVerticalCenterY(data.pageConfig.componentKey);
          let parentRectY = Utils.getRectVerticalCenterY(data.pageConfig.parentComponentKey);
          console.info('[' + data.caseTag + '] rectY : ' + rectY);
          console.info('[' + data.caseTag + '] parentRectY : ' + parentRectY);
          expect(rectY).assertEqual(parentRectY);
          expect(Math.floor(Math.abs(rect.left - parentRect.left)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetEnd') {
          //When alignSelf : ItemAlign.End
          expect(Math.floor(Math.abs(rect.bottom - parentRect.bottom)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.left - parentRect.left)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetStretch') {
          //When alignSelf : ItemAlign.Stretch
          console.info('[' + data.caseTag + '] rect.bottom - rect.top: ' + (rect.bottom - rect.top));
          console.info('[' + data.caseTag + '] vp2px(parentHeight): ' + vp2px(parentHeight));
          if (data.pageConfig.componentKey == 'TextInput' || data.pageConfig.componentKey == 'TextArea') {
            expect(Math.floor(Math.abs((rect.bottom - rect.top)- vp2px(parentHeight))) <= 2).assertTrue();
          } else {
            expect(Math.floor(Math.abs((rect.bottom - rect.top)- vp2px(parentHeight))) <= 1).assertTrue();
          }
        }
      } else if (data.pageConfig.containerView == 'Column' || data.pageConfig.containerView == 'FlexColumn') {
        if (data.testValue.describe == 'SetAuto' || data.testValue.describe == 'SetStart') {
          //When alignSelf : ItemAlign.Auto or ItemAlign.Start
          expect(Math.floor(Math.abs(rect.top - parentRect.top)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.left - parentRect.left)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetCenter') {
          //When alignSelf : ItemAlign.Center
          let rectX = Utils.getRectHorizontalCenterX(data.pageConfig.componentKey);
          let parentRectX = Utils.getRectHorizontalCenterX(data.pageConfig.parentComponentKey);
          console.info('[' + data.caseTag + '] rectX : ' + rectX);
          console.info('[' + data.caseTag + '] parentRectX : ' + parentRectX);
          expect(rectX).assertEqual(parentRectX);
          expect(Math.floor(Math.abs(rect.top - parentRect.top)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetEnd') {
          //When alignSelf : ItemAlign.End
          expect(Math.floor(Math.abs(rect.top - parentRect.top)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.right - parentRect.right)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetStretch') {
          //When alignSelf : ItemAlign.Stretch
          console.info('[' + data.caseTag + '] rect.right - rect.left: ' + (rect.right - rect.left));
          console.info('[' + data.caseTag + '] vp2px(parentWidth): ' + vp2px(parentWidth));
          if (data.pageConfig.componentKey == 'TextInput' || data.pageConfig.componentKey == 'TextArea') {
            expect(Math.floor(Math.abs((rect.right - rect.left)- vp2px(parentWidth))) <= 2).assertTrue();
          } else {
            expect(Math.floor(Math.abs((rect.right - rect.left)- vp2px(parentWidth))) <= 1).assertTrue();
          }
        }
      } else if (data.pageConfig.containerView == 'FlexRowReverse') {
        //Special parent component
        if (data.testValue.describe == 'SetAuto' || data.testValue.describe == 'SetStart') {
          //When alignSelf : ItemAlign.Auto or ItemAlign.Start
          expect(Math.floor(Math.abs(rect.top - parentRect.top)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.right - parentRect.right)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetCenter') {
          //When alignSelf : ItemAlign.Center
          let rectY = Utils.getRectVerticalCenterY(data.pageConfig.componentKey);
          let parentRectY = Utils.getRectVerticalCenterY(data.pageConfig.parentComponentKey);
          console.info('[' + data.caseTag + '] rectY : ' + rectY);
          console.info('[' + data.caseTag + '] parentRectY : ' + parentRectY);
          expect(rectY).assertEqual(parentRectY);
          expect(Math.floor(Math.abs(rect.right - parentRect.right)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetEnd') {
          //When alignSelf : ItemAlign.End
          expect(Math.floor(Math.abs(rect.bottom - parentRect.bottom)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.right - parentRect.right)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetStretch') {
          //When alignSelf : ItemAlign.Stretch
          console.info('[' + data.caseTag + '] rect.bottom - rect.top: ' + (rect.bottom - rect.top));
          console.info('[' + data.caseTag + '] vp2px(parentHeight): ' + vp2px(parentHeight));
          if (data.pageConfig.componentKey == 'TextInput' || data.pageConfig.componentKey == 'TextArea') {
            expect(Math.floor(Math.abs((rect.bottom - rect.top)- vp2px(parentHeight))) <= 2).assertTrue();
          } else {
            expect(Math.floor(Math.abs((rect.bottom - rect.top)- vp2px(parentHeight))) <= 1).assertTrue();
          }
        }
      } else if (data.pageConfig.containerView == 'FlexColumnReverse') {
        //Special parent component
        if (data.testValue.describe == 'SetAuto' || data.testValue.describe == 'SetStart') {
          //When alignSelf : ItemAlign.Auto or ItemAlign.Start
          expect(Math.floor(Math.abs(rect.bottom - parentRect.bottom)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.left - parentRect.left)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetCenter') {
          //When alignSelf : ItemAlign.Center
          let rectX = Utils.getRectHorizontalCenterX(data.pageConfig.componentKey);
          let parentRectX = Utils.getRectHorizontalCenterX(data.pageConfig.parentComponentKey);
          console.info('[' + data.caseTag + '] rectX : ' + rectX);
          console.info('[' + data.caseTag + '] parentRectX : ' + parentRectX);
          expect(rectX).assertEqual(parentRectX);
          expect(Math.floor(Math.abs(rect.bottom - parentRect.bottom)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetEnd') {
          //When alignSelf : ItemAlign.End
          expect(Math.floor(Math.abs(rect.bottom - parentRect.bottom)) <= 1).assertTrue();
          expect(Math.floor(Math.abs(rect.right - parentRect.right)) <= 1).assertTrue();
        } else if (data.testValue.describe == 'SetStretch') {
          //When alignSelf : ItemAlign.Stretch
          console.info('[' + data.caseTag + '] rect.right - rect.left: ' + (rect.right - rect.left));
          console.info('[' + data.caseTag + '] vp2px(parentWidth): ' + vp2px(parentWidth));
          if (data.pageConfig.componentKey == 'TextInput' || data.pageConfig.componentKey == 'TextArea') {
            expect(Math.floor(Math.abs((rect.right - rect.left)- vp2px(parentWidth))) <= 2).assertTrue();
          } else {
            expect(Math.floor(Math.abs((rect.right - rect.left)- vp2px(parentWidth))) <= 1).assertTrue();
          }
        }
      }
    })
  })
}