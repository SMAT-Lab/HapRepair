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
  compareWidth: number = 0;
  compareHeight: number = 0;
  parentComponentKey: string = '';
  referenceComponentKey: string = '';
}

class TestValue {
  describe: string = '';
  setValue: SetValue = new SetValue();
  expectValue: ESObject;
}

class SetValue {
  direction: FlexDirection = FlexDirection.Row;
  alignSelf: ItemAlign = ItemAlign.Auto;
  wrap: FlexWrap = FlexWrap.NoWrap;
}

export default function alignSelfStretchNoSizeTest() {

  let supportView = [
    'AlphabetIndexer',
    'Blank',
    'Button',
    'Checkbox',
    'CheckboxGroup',
    'DataPanel',
    'DatePicker',
    'Divider',
    'Gauge',
    'Image',
    'ImageAnimator',
    'LoadingProgress',
    'Marquee',
    'Menu',
    'Navigation',
    'NavRouter',
    'ScrollBar',
    'RowSplit',
    'Row',
    'RelativeContainer',
    'Navigator',
    'ColumnSplit',
    'TextTimer',
    'Stepper',
    'Flex',
    'Progress',
    'QRCode',
    'Radio',
    'Rating',
    'Search',
    'Select',
    'Slider',
    'Text',
    'TextArea',
    'TextClock',
    'TextInput',
    'TextPicker',
    'TimePicker',
    'Toggle',
    'Badge',
    'Column',
    'Counter',
    'GridRow',
    'Grid',
    'List',
    'Panel',
    'Refresh',
    'Scroll',
    'SideBarContainer',
    'Stack',
    'Swiper',
    'Tabs',
    'WaterFlow',
    'Circle',
    'Ellipse',
    'Line',
    'Polyline',
    'Polygon',
    'Path',
    'Rect',
    'Shape'
  ]

  let containerView = ['Flex', 'FlexNoReference']

  let parentWidth = 410;
  let parentHeight = 610;
  let compareWidth = 210;
  let compareHeight = 210;

  let pageConfig: PageConfig = {
    testName: 'testAlignSelf',
    pageName: 'AlignSelfStretchNoSizePage',
    pageUrl: 'MainAbility/pages/flex/AlignSelfStretchNoSizePage',
    parentWidth: parentWidth,
    parentHeight: parentHeight,
    compareWidth: compareWidth,
    compareHeight: compareHeight,
    parentComponentKey: 'parentComponentKey',
    referenceComponentKey: 'referenceComponentKey'
  }

  let testValues: TestValue[] = [{
    describe: 'SetStretchRowWrap',
    setValue: {
      direction: FlexDirection.Row,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.Wrap
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetStretchRowWrapReverse',
    setValue: {
      direction: FlexDirection.Row,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.WrapReverse
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetStretchRowReverseWrap',
    setValue: {
      direction: FlexDirection.RowReverse,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.Wrap
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetStretchRowReverseWrapReverse',
    setValue: {
      direction: FlexDirection.RowReverse,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.WrapReverse
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetStretchColumnWrap',
    setValue: {
      direction: FlexDirection.Column,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.Wrap
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetStretchColumnWrapReverse',
    setValue: {
      direction: FlexDirection.Column,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.WrapReverse
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetStretchColumnReverseWrap',
    setValue: {
      direction: FlexDirection.ColumnReverse,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.Wrap
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }, {
    describe: 'SetStretchColumnReverseWrapReverse',
    setValue: {
      direction: FlexDirection.ColumnReverse,
      alignSelf: ItemAlign.Stretch,
      wrap: FlexWrap.WrapReverse
    },
    expectValue: {
      expectAlignSelf: 'ItemAlign.Stretch'
    }
  }]

  describe('AlignSelfStretchNoSizeTest', () => {
    CommonTest.initTestByContainerView(pageConfig, containerView, supportView, testValues, async (data: ESObject) => {

      //Get component rect and parent component rect
      let parentRect = Utils.getComponentRect(data.pageConfig.parentComponentKey);
      let rect = Utils.getComponentRect(data.pageConfig.componentKey);

      //Judge attribute
      expect(data.viewObj.$attrs.alignSelf).assertEqual(data.testValue.expectValue.expectAlignSelf);

      //When set different value, judge the alignment is correct
      if ('Flex' == data.pageConfig.containerView) {
        let referenceComponentRect = Utils.getComponentRect(data.pageConfig.referenceComponentKey);
        if ((data.testValue.describe == 'SetStretchRowWrap')
          || (data.testValue.describe == 'SetStretchRowReverseWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.left).assertEqual(parentRect.left);
          expect(rect.top).assertEqual(parentRect.top);
          expect(rect.bottom - rect.top).assertEqual(referenceComponentRect.bottom - referenceComponentRect.top);
        } else if ((data.testValue.describe == 'SetStretchRowReverseWrap')
          || (data.testValue.describe == 'SetStretchRowWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.right).assertEqual(parentRect.right);
          expect(rect.top).assertEqual(parentRect.top);
          expect(rect.bottom - rect.top).assertEqual(referenceComponentRect.bottom - referenceComponentRect.top);
        } else if ((data.testValue.describe == 'SetStretchColumnWrap')
          || (data.testValue.describe == 'SetStretchColumnReverseWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.left).assertEqual(parentRect.left);
          expect(rect.top).assertEqual(parentRect.top);
          expect(rect.right - rect.left).assertEqual(referenceComponentRect.right - referenceComponentRect.left);
        } else if ((data.testValue.describe == 'SetStretchColumnReverseWrap')
          || (data.testValue.describe == 'SetStretchColumnWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.bottom).assertEqual(parentRect.bottom);
          expect(rect.left).assertEqual(parentRect.left);
          expect(rect.right - rect.left).assertEqual(referenceComponentRect.right - referenceComponentRect.left);
        }
      } else if ('FlexNoReference' == data.pageConfig.containerView) {
        if ((data.testValue.describe == 'SetStretchRowWrap')
          || (data.testValue.describe == 'SetStretchRowReverseWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.left).assertEqual(parentRect.left);
          expect(rect.top).assertEqual(parentRect.top);
          expect(rect.bottom - rect.top).assertEqual(vp2px(parentHeight));
        } else if ((data.testValue.describe == 'SetStretchRowReverseWrap')
          || (data.testValue.describe == 'SetStretchRowWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.right).assertEqual(parentRect.right);
          expect(rect.top).assertEqual(parentRect.top);
          expect(rect.bottom - rect.top).assertEqual(vp2px(parentHeight));
        } else if ((data.testValue.describe == 'SetStretchColumnWrap')
          || (data.testValue.describe == 'SetStretchColumnReverseWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.left).assertEqual(parentRect.left);
          expect(rect.top).assertEqual(parentRect.top);
          expect(rect.right - rect.left).assertEqual(vp2px(parentWidth));
        } else if ((data.testValue.describe == 'SetStretchColumnReverseWrap')
          || (data.testValue.describe == 'SetStretchColumnWrapReverse')) {
          //Determine if alignSelf will override the alignItems setting of the parent container
          //And control the alignment of the component
          expect(rect.bottom).assertEqual(parentRect.bottom);
          expect(rect.left).assertEqual(parentRect.left);
          expect(rect.right - rect.left).assertEqual(vp2px(parentWidth));
        }
      }
    })
  })
}