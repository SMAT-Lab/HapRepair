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

import { assert, describe, it } from 'vitest';
import { SceneConfig } from '../../src/Config';
import { Scene } from '../../src/Scene';

describe('build scene by files Test', () => {

    const filesPath: string[] = ['/'];
    const projectDir = './tests/resources/dependency/exampleProject/MyApplication4Files';
    const projectName = 'MyApplication4Files';
    const sceneConfig: SceneConfig = new SceneConfig();
    sceneConfig.buildFromProjectFiles(projectName, projectDir, filesPath);

    const scene: Scene = new Scene();
    scene.buildSceneFromFiles(sceneConfig);
    scene.inferTypes();
    it('check files num', () => {
        assert.equal(scene.getFiles().length, 17);
        assert.equal(scene.getModuleSceneMap().size, 6);
        assert.equal(scene.getClasses().length, 28);
        assert.equal(scene.getMethods().length, 64);
        assert.equal(scene.getModuleSceneMap().size, 6);
        assert.equal(scene.getModuleSceneMap().get('lib1')?.getModuleFilesMap().size, 2);
        assert.equal(scene.getModuleSceneMap().get('libbase')?.getModuleFilesMap().size, 4);
        assert.equal(scene.getModuleSceneMap().get('log4js')?.getModuleFilesMap().size, 2);
        assert.equal(scene.getModuleSceneMap().get('model1')?.getModuleFilesMap().size, 4);
        assert.equal(scene.getModuleSceneMap().get('model2')?.getModuleFilesMap().size, 2);
        assert.equal(scene.getModuleSceneMap().get('parameter_test')?.getModuleFilesMap().size, 1);
    });

    it('check dependencies info', () => {
        assert.equal(scene.getOverRides().get('@model1'), './model1/index11.ets');
        assert.equal(scene.getOverRides().get('@model2'), 'file:./model2');
        assert.equal(JSON.stringify(scene.getGlobalModule2PathMapping()!), JSON.stringify(globalModule2PathMapping_expect_result));
        assert.equal(scene.getbaseUrl(), './');
    });

});



export const globalModule2PathMapping_expect_result = {
    '@DependencyTest/': [
        '../DependencyTest/',
        '../DependencyTest1/'
    ],
    '@DependencyTest/*': [
        '../DependencyTest/*',
        '../DependencyTest1/*'
    ],
    'DependencyTest/': [
        '../DependencyTest/',
        '../DependencyTest1/'
    ],
    'DependencyTest/*': [
        '../DependencyTest/*',
        '../DependencyTest1/*'
    ]
};

