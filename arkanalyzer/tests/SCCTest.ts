/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use this file except in compliance with the License.
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
import { CallGraph, CallGraphNode } from '../src/callgraph/model/CallGraph';
import { CallGraphBuilder } from '../src/callgraph/model/builder/CallGraphBuilder';
import Logger, { LOG_LEVEL } from '../src/utils/logger';
import { SCCDetection } from '../src/core/graph/Scc';

Logger.configure('./out/ArkAnalyzer.log', LOG_LEVEL.TRACE);
let config: SceneConfig = new SceneConfig();

function runDir(): CallGraph {
    config.buildFromProjectDir('./tests/resources/scc');

    let projectScene: Scene = new Scene();
    projectScene.buildSceneFromProjectDir(config);
    projectScene.inferTypes();

    let cg = new CallGraph(projectScene);
    let cgBuilder = new CallGraphBuilder(cg, projectScene);
    cgBuilder.buildDirectCallGraphForScene();
    cg.dump('out/scccg.dot');
    return cg;
}

let cg = runDir();
let scc = new SCCDetection<CallGraph>(cg);
scc.find();

let topo = scc.getTopoAndCollapsedNodeStack();
console.log(topo);
while (topo.length > 0) {
    let n = topo.pop()!;

    let f: CallGraphNode = cg.getNode(n) as CallGraphNode;
    console.log(f.getMethod().getMethodSubSignature().getMethodName());
    let subn = scc.getSubNodes(n);
    subn.forEach(s => {
        let f: CallGraphNode = cg.getNode(s) as CallGraphNode;
        console.log('  ' + f.getMethod().getMethodSubSignature().getMethodName());
    });
}

console.log('===\n');
for (let n of cg.getNodesIter()) {
    let sccNodes = scc.getMySCCNodes(n.getID());
    console.log((n as CallGraphNode).getMethod().getMethodSubSignature().getMethodName());

    for (let sn of sccNodes) {
        console.log('  ' + (cg.getNode(sn) as CallGraphNode).getMethod().getMethodSubSignature().getMethodName());
    }
}