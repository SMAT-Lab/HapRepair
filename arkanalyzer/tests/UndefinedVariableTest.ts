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
import { UndefinedVariableChecker, UndefinedVariableSolver } from '../src/core/dataflow/UndefinedVariable';

// const config_path = "tests\\resources\\ifds\\UndefinedVariable\\ifdsTestConfig.json";
let config: SceneConfig = new SceneConfig();
config.buildFromProjectDir("tests/resources/ifds/UndefinedVariable");
const scene = new Scene();
scene.buildBasicInfo(config);
scene.buildSceneFromProjectDir(config);
const defaultMethod = scene.getFiles()[0].getDefaultClass().getDefaultArkMethod();
let method = ModelUtils.getMethodWithName("U2",defaultMethod!);
if(method){
    const problem = new UndefinedVariableChecker([...method.getCfg()!.getBlocks()][0].getStmts()[method.getParameters().length],method);
    const solver = new UndefinedVariableSolver(problem, scene);
    solver.solve();
    for (const outcome of problem.getOutcomes()) {
        let position = outcome.stmt.getOriginPositionInfo();
        console.log('undefined error in line ' + position.getLineNo());
    }
    debugger;
}
