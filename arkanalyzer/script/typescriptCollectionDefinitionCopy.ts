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

import * as fs from 'fs';
import * as path from 'path';

const targetDir = './builtIn/typescript';

function emptyDir(dir: string): void {
  if (fs.existsSync(dir)) {
    fs.readdirSync(dir).forEach(file => {
      const filePath = path.join(dir, file);
      if (fs.statSync(filePath).isFile()) {
        fs.unlinkSync(filePath);
      } else if (fs.statSync(filePath).isDirectory()) {
        emptyDir(filePath);
        fs.rmdirSync(filePath);
      }
    });
  }
}

if (!fs.existsSync(targetDir)) {
  fs.mkdirSync(targetDir, { recursive: true });
} else {
  emptyDir(targetDir);
}

fs.mkdirSync(path.join(targetDir, 'api'));
fs.mkdirSync(path.join(targetDir, 'api/@internal'));

const sourceFile1 = './node_modules/typescript/lib/lib.es2015.collection.d.ts';
const sourceFile2 = './node_modules/typescript/lib/lib.es5.d.ts';

function copyFile(source: string, destination: string): void {
  if (fs.existsSync(source)) {
    fs.copyFileSync(source, destination);
  } else {
    console.log('Error copy typescript collection definition file');
  }
}

copyFile(sourceFile1, path.join(targetDir, 'api/@internal', 'lib.es2015.collection.d.ts'));
copyFile(sourceFile2, path.join(targetDir, 'api/@internal', 'lib.es5.d.ts'));