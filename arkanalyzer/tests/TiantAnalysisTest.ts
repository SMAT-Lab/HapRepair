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
import { ModelUtils } from '../src/core/common/ModelUtils';
import { TiantAnalysisChecker, TiantAnalysisSolver } from '../src/core/dataflow/TiantAnalysis';

const config_path = "tests\\resources\\ifds\\TiantAnalysis\\ifdsTestConfig.json";
let config: SceneConfig = new SceneConfig();
config.buildFromJson2(config_path);
const scene = new Scene(config);
scene.inferTypes()
const defaultMethod = scene.getFiles()[0].getDefaultClass().getDefaultArkMethod();
let method = ModelUtils.getMethodWithName("T1",defaultMethod!);
let source = ModelUtils.getMethodWithName("source",defaultMethod!);
let sink = ModelUtils.getMethodWithName("sink",defaultMethod!);
if(method){
    const problem = new TiantAnalysisChecker([...method.getCfg().getBlocks()][0].getStmts()[method.getParameters().length],method);
    problem.setSinks([sink!]);
    problem.setSources([source!]);
    const solver = new TiantAnalysisSolver(problem, scene);
    solver.solve();
}
