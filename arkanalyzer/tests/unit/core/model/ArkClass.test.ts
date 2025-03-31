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

import { assert, describe, expect, it } from 'vitest';
import { ArkClass, ArkMethod, ClassSignature, MethodSignature, Scene, SceneConfig } from '../../../../src';
import path from 'path';

let config: SceneConfig = new SceneConfig();
config.buildFromProjectDir(path.join(__dirname, '../../../resources/model/class'));
let projectScene: Scene = new Scene();
projectScene.buildSceneFromProjectDir(config);
projectScene.inferTypes();

let arkFile = projectScene.getFiles().find((file) => file.getName() === 'class.ts');
let arkClass = arkFile?.getClassWithName('TestClass');
let arkMethod = arkClass?.getMethodWithName('testMethod');

describe('ArkClass Test', () => {
    it('get method with matched signature', async () => {
        assert.isNotNull(arkClass);
        assert.isDefined(arkMethod);
        assert.isNotNull(arkMethod);
        const matchedSignature = new MethodSignature((arkClass as ArkClass).getSignature(), (arkMethod as ArkMethod).getSubSignature());
        const method = arkClass?.getMethod(matchedSignature);
        assert.isDefined(method);
        assert.isNotNull(method);
        expect(method?.getLine()).toEqual(19);
    });

    it('get method with unmatched signature', async () => {
        assert.isNotNull(arkClass);
        assert.isDefined(arkMethod);
        assert.isNotNull(arkMethod);
        const clsSignature = new ClassSignature('newClass', (arkClass as ArkClass).getDeclaringArkFile().getFileSignature());
        const unmatchedSignature = new MethodSignature(clsSignature, (arkMethod as ArkMethod).getSubSignature());
        const method = arkClass?.getMethod(unmatchedSignature);
        assert.isNull(method);
    });
});