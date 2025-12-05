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

import InterceptionTest from './Interception/InterceptionTest.test'
import NavigationTest from './Navigation/NavigationTest.test'
import EnableShowHideWithContentCoverPage from './Navigation/EnableShowHideWithContentCoverPage.test'
import SplitPlaceholder from './Navigation/SplitPlaceholder.test'
import UIContextNavDestinationUpdate from './UIContext/UIContextNavDestinationUpdate.test'
import StepperModifierPage from './Stepper/StepperModifierPage.test'
import HomePage from './Navigation/HomePage.test'
import StepperModifierTest from './Stepper/StepperModifierTest.test'
import NavigatorTargetJsunit from './Navigator/navigator.test'
import PageInfo1 from './UIContext/PageInfo1.test'
import NavigationAdd from './Navigation/NavigationAdd.test'

import NavigationTitleTextModifier from './Navigation/NavigationTitleTextModifier.test'
import EntryOptionPage2 from './Navigation/EntryOptionPage2.test'
import EntryOptionPage from './Navigation/EntryOptionPage.test'
import NavigationCustomTitlePage from './Navigation/NavigationCustomTitlePage.test'
import TitleOptionsToolOptions from './Navigation/TitleOptionsToolOptions.test'
import BarStylePage from './Navigation/BarStylePage.test'
import HideToolBarNavPage from './Navigation/HideToolBarNavPage.test'
import NavDestinationUIObserver from './Navigation/NavDestinationUIObserver.test'

import SafeAreaPadding from './BarStyleTest/SafeAreaPadding.test'
import NavigationEnableDragBar from './EnableDragBarTest/NavigationEnableDragBar.test'
import NavigationBindToScrollable from './BindToScrollable/NavigationBindToScrollable.test'
import NavigationBindToNestedScrollable from './BindToScrollable/NavigationBindToNestedScrollable.test'
import Recoverable from './Navigation/Recoverable.test'
import SystemTransition from './Navigation/SystemTransition.test'
import RouterRecoverable from './Navigation/RouterRecoverable.test'
import RouterRecoverable2 from './Navigation/RouterRecoverable2.test'
import SystemTransition2 from './Navigation/SystemTransition2.test'

import EnableModeChangeAnimation from './Navigation/EnableModeChangeAnimation.test';
import OnResultPage from './Navigation/OnResultPage.test';
import CustomTransitionPage from './NavDestination/CustomTransitionPage.test';
import HideBackButtonPage from './NavDestination/HideBackButtonPage.test';
import NavDestinationSystemTransition from './NavDestination/SystemTransition2.test';

import ReplaceDestination1Test from './Navigation/ReplaceDestination1.test'
import ReplaceDestination2 from './Navigation/ReplaceDestination2.test'
import QueryRouterPageInfoUndefined from './Navigation/QueryRouterPageInfoUndefined.test'
import PushDestination401 from './Navigation/PushDestination401.test'

import OnActivePage from './OnActivePage.test'
import PathStackPage from './PathStackPage.test'
import NavDestinationStateActive from './NavDestinationStateActive.test'

import Navigation1 from './CustomNavigation/Navigation1.test'
import Navigation2 from './CustomNavigation/Navigation2.test'
import Navigation3 from './CustomNavigation/Navigation3.test'
import Navigation4 from './CustomNavigation/Navigation4.test'

export default function testsuite() {
  PathStackPage();

  InterceptionTest();
  NavigationTest();
  EnableShowHideWithContentCoverPage();
  SplitPlaceholder();
  UIContextNavDestinationUpdate();
  StepperModifierPage();
  HomePage();
  StepperModifierTest();
  NavigatorTargetJsunit();
  PageInfo1();
  NavigationAdd();

  NavigationTitleTextModifier();
  EntryOptionPage();
  EntryOptionPage2();
  NavigationCustomTitlePage();
  TitleOptionsToolOptions();
  BarStylePage();
  HideToolBarNavPage();
  NavDestinationUIObserver();

  SafeAreaPadding();
  NavigationEnableDragBar();
  NavigationBindToScrollable();
  NavigationBindToNestedScrollable();
  Recoverable();
  SystemTransition();
  RouterRecoverable();
  SystemTransition2();

  EnableModeChangeAnimation();
  OnResultPage();
  CustomTransitionPage();
  HideBackButtonPage();
  NavDestinationSystemTransition();

  ReplaceDestination1Test();
  ReplaceDestination2();
  QueryRouterPageInfoUndefined();
  PushDestination401();

  OnActivePage();
  NavDestinationStateActive();
  Navigation1()
  Navigation2()
  Navigation3()
  Navigation4()
} 