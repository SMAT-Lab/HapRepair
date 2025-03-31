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

import { SceneConfig } from "../src/Config";
import { DummyMainCreater } from "../src/core/common/DummyMainCreater";
import { Scene } from "../src/Scene";


const config_path = "tests\\resources\\ifds\\TiantAnalysis\\ifdsTestConfig.json";
// const config_path = "tests\\resources\\typeInference\\ohapps.json";
let config: SceneConfig = new SceneConfig();
config.buildFromJson(config_path);
// config.buildFromProjectDir("tests/resources")
const scene = new Scene();
scene.buildBasicInfo(config);
scene.buildScene4HarmonyProject();
// scene.buildSceneFromProjectDir(config);
scene.inferTypes()
const creater = new DummyMainCreater(scene);
creater.createDummyMain();
let d = creater.getDummyMain()
console.log(d)