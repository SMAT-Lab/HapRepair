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

import { SceneConfig } from '../../src/Config';
import { assert, describe, it } from 'vitest';

describe('SceneConfig Test', () => {
    it('true case', () => {
        let config: SceneConfig = new SceneConfig();
        config.buildFromJson('./tests/resources/scene/SceneTestConfig.json');
        assert.equal(config.getTargetProjectName(), 'applications_photos');
        assert.equal(config.getTargetProjectDirectory(), 'D:\\Codes\\resources\\applications\\applications_photos');
        assert.equal(config.getSdksObj().length, 1);
    })

    it('config file does not exit case', () => {
        let config: SceneConfig = new SceneConfig();
        config.buildFromJson('./tests/resources/scene/NotExist.json');
        assert.equal(config.getTargetProjectName(), '');
        assert.equal(config.getTargetProjectDirectory(), '');
        assert.equal(config.getSdksObj().length, 0);
    })

    it('test', () => {
        let a: Map<string, number> = new Map<string, number>();
        a.set('c', 0);
        a.set('i', 1);
        a.set('m', 2);
        a.set('c', 4);
        assert.equal(a.size, 3);
    })
})