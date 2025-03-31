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

import { SceneConfig } from '../src/Config';
import { Scene } from '../src/Scene';
import { PrinterBuilder } from '../src/save/PrinterBuilder';
import { join } from 'path';
import Logger, { LOG_LEVEL, LOG_MODULE_TYPE } from '../src/utils/logger';

const logPath = 'out/ArkAnalyzer.log';
const logger = Logger.getLogger(LOG_MODULE_TYPE.TOOL, 'SceneTest');
Logger.configure(logPath, LOG_LEVEL.DEBUG, LOG_LEVEL.DEBUG);

function testAppProjectSave() {
    let config: SceneConfig = new SceneConfig();
    config.buildFromJson(join(__dirname, './AppTestConfig.json'));
    let scene: Scene = new Scene();
    scene.buildBasicInfo(config);
    logger.error('start ... ');
    scene.buildScene4HarmonyProject();
    scene.inferTypes();
    logger.error('end inferTypes ... ');

    for (let cls of scene.getClasses()) {
        if (cls.hasComponentDecorator()) {
            cls.getViewTree();
        }
    }
    logger.error('end build viewtree ... ');

    let printer: PrinterBuilder = new PrinterBuilder(join(__dirname, '..', 'out/project'));
    for (let f of scene.getFiles()) {
        printer.dumpToTs(f);
    }
}

function testSimpleSave() {
    let config: SceneConfig = new SceneConfig();
    config.buildFromProjectDir(join(__dirname, 'resources', 'save'));
    let scene: Scene = new Scene();
    scene.buildSceneFromProjectDir(config);
    let printer: PrinterBuilder = new PrinterBuilder(join(__dirname, '..', 'out/save'));
    for (let f of scene.getFiles()) {
        printer.dumpToTs(f);
    }
}

testAppProjectSave();
testSimpleSave();