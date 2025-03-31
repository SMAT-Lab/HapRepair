import { SceneConfig } from "../src/Config";
import { Scene } from "../src/Scene";
import { PrinterBuilder } from '../src';
import path from "path";

let config: SceneConfig = new SceneConfig();
let projectName = 'ts_files'
config.buildFromProjectDir(path.join(__dirname, '../tests/projects/' + projectName));
let scene = new Scene();
scene.buildSceneFromProjectDir(config);
scene.inferTypes();

const printerBuilder = new PrinterBuilder()
for (const arkFile of scene.getFiles()) {
    printerBuilder.dumpToDot(arkFile, 'ts_cfg/' + arkFile.getName() + '.dot')
}
