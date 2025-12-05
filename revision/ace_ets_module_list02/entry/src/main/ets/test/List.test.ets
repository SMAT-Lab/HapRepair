/**
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
import listJsunit from './List/ListJsunit.test';
import ListDeleteTest from './List/ListDelete.test'
import ListSwipeTest from './List/ListSwipe.test'
import ListScrollTest from './List/ListScroll.test'
import listSContentTest from "./List/ListSContent.test"
import edgeEffectJsunit from './List/EdgeEffectJsunit.test'
import listJsunit1 from './List/ListJsunit1.test';
import listItemJsunit from './ListItem/ListItemJsunit.test';
import ItemOnSelectJsunit from './ListItem/listItem.test'
import listItemJsunit2 from './ListItem/listItemJsunit2.test';
import List07Test from './List/List07.test';
import listIdleJsunit from './List/listIdleJsunit.test';
import listNewJsunit from './List/ListNewTest.test';
import listScrollTo from './List/listScrollTo.test';
import focusedList from './List/focusedList.test'
import focusedCardList from './ListItemGroup/focusedCardList.test'
import ListScrollToItemInGroup from './ListItemGroup/ListScrollToItemInGroup.test'
import listUxTest from './ListItemGroup/listux.test'
import conSizeGroup1 from './ListItemGroup/conSizeGroup1.test';
import conSizeGroup2 from './ListItemGroup/conSizeGroup2.test';
import conSizeGroup3 from './ListItemGroup/conSizeGroup3.test'
import lanesHeaderGroup from './ListItemGroup/lanesHeaderGroup.test'
import lanesItemGroup from './ListItemGroup/lanesItemGroup.test';
import singleHozList from './ListItemGroup/singleHozList.test'
import singleHozDiff from './ListItemGroup/singleHozDiff.test'
import listVisibleJsunit from './ListItemGroup/listVisibleJsunit.test';
import ListItemGroupNone from './ListItemGroup/listItemGroupNone.test'
import ListMainChildrenTest from './ListItemGroup/ListMainChildren.test';
import paddingStartOffset from './contentOffset/paddingStartOffset.test';
import headerStartOffset from './contentOffset/headerStartOffset.test';
import toIndexStartOffset from './contentOffset/toIndexStartOffset.test';
import offsetEdgeBtm from './contentOffset/offsetEdgeBtm.test';
import offsetEdgeTop from './contentOffset/offsetEdgeTop.test';
import offsetAlignEnd from './contentOffset/offsetAlignEnd.test';
import offsetAlignStart from './contentOffset/offsetAlignStart.test';
import childrenSizeError from './errorCode/childrenSizeError.test';
import flingError from './errorCode/flingError.test';
import setTest from './childrenMainSize/setTest.test'
import getTest from './childrenMainSize/getTest.test'
import ListWillScroll from './List/ListWillScroll.test';
import offsetRemain from './List/offsetRemain.test';
import listLevel1Test from './List/ListLevel1.test';
import listLevel1HorizontalTest from './List/ListLevel1Horizontal.test';
import listSwipeActionApi12Test from './List/listSwipeActionApi12.test';
import chainAnimationList from './List/chainAnimationList.test'
import ListAddApi from './List/ListAddApi.test';
import listDidJsunit from './List/ListDidJsunit.test';
import listOffsetJsunit from './List/listOffsetJsunit.test';
import listVisibleContent from './List/listVisibleContent.test';
import ListFlingCallback from './List/ListFlingCallback.test'
import ListScrollEdgeCallback from './List/ListScrollEdgeCallback.test';

export default function testsuite() {
  listJsunit()
  ListDeleteTest();
  ListSwipeTest();
  ListScrollTest();
  listSContentTest();
  edgeEffectJsunit();
  listJsunit1()
  listItemJsunit()
  ItemOnSelectJsunit();
  listItemJsunit2();
  List07Test();
  listUxTest();
  listIdleJsunit();
  listNewJsunit();
  listScrollTo()
  singleHozList()
  singleHozDiff()
  focusedList()
  focusedCardList()
  ListScrollToItemInGroup()
  conSizeGroup1()
  conSizeGroup2()
  conSizeGroup3()
  lanesHeaderGroup()
  lanesItemGroup()
  paddingStartOffset();
  headerStartOffset();
  toIndexStartOffset();
  offsetEdgeBtm();
  offsetEdgeTop();
  offsetAlignEnd();
  offsetAlignStart();
  ListWillScroll()
  offsetRemain()
  listLevel1Test()
  listLevel1HorizontalTest()
  listSwipeActionApi12Test()
  chainAnimationList()
  childrenSizeError()
  flingError()
  setTest()
  getTest()
  ListAddApi()
  listOffsetJsunit()
  listVisibleContent()
  listDidJsunit()
  ListFlingCallback()
  ListScrollEdgeCallback()
}