/**
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
import windowAnimationManager from '@ohos.animation.windowAnimationManager';

export default class WindowAnimationControllerImpl implements windowAnimationManager.WindowAnimationController {
  onStartAppFromLauncher(startingWindowTarget: windowAnimationManager.WindowAnimationTarget,
    finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback): void {
    console.log(`remote window animaion onStartAppFromLauncher`);
    finishedCallback.onAnimationFinish();
  }

  onStartAppFromRecent(startingWindowTarget: windowAnimationManager.WindowAnimationTarget,
    finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback): void {
    console.log(`remote window animaion onStartAppFromRecent`);
    finishedCallback.onAnimationFinish();
  }

  onStartAppFromOther(startingWindowTarget: windowAnimationManager.WindowAnimationTarget,
    finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback): void {
    console.log(`remote window animaion onStartAppFromOther`);
    finishedCallback.onAnimationFinish();
  }

  onAppTransition(fromWindowTarget: windowAnimationManager.WindowAnimationTarget,
    toWindowTarget: windowAnimationManager.WindowAnimationTarget,
    finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback): void {
    console.log(`remote window animaion onAppTransition`);
    finishedCallback.onAnimationFinish();
  }

  onMinimizeWindow(minimizingWindowTarget: windowAnimationManager.WindowAnimationTarget,
    finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback): void {
    console.log(`remote window animaion onMinimizeWindow`);
    finishedCallback.onAnimationFinish();
  }

  onCloseWindow(closingWindowTarget: windowAnimationManager.WindowAnimationTarget,
    finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback): void {
    console.log(`remote window animaion onCloseWindow`);
    finishedCallback.onAnimationFinish();
  }

  onScreenUnlock(finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback): void {
    console.log(`remote window animaion onScreenUnlock`);
    finishedCallback.onAnimationFinish();
  }

  onWindowAnimationTargetsUpdate(fullScreenWindowTarget: windowAnimationManager.WindowAnimationTarget,
    floatingWindowTargets: Array<windowAnimationManager.WindowAnimationTarget>): void {
    console.log('onWindowAnimationTargetsUpdate, the fullScreenWindowTarget is: ' + fullScreenWindowTarget);
    console.log('onWindowAnimationTargetsUpdate, the floatingWindowTargets are: ' + floatingWindowTargets);
  }
}