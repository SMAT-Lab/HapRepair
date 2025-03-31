/************************
 * main.ts
 ************************/

import path from "path";
import fs from "fs";

/**
 * 这里用到的类型定义都来自你给出的文件：
 * - ArkFile / ArkClass / ArkMethod / ArkBody / Cfg / BasicBlock / Stmt
 * - 以及 Scene / SceneConfig / PrinterBuilder
 * 请根据你项目中实际的 import 路径调整下面这些 import。
 */
import { SceneConfig } from "./src/Config";
import { Scene } from "./src/Scene";
import { ArkFile } from "./src/core/model/ArkFile";
import { ArkClass } from "./src/core/model/ArkClass";
import { ArkMethod } from "./src/core/model/ArkMethod";
import { Stmt } from "./src/core/base/Stmt";
import { BasicBlock } from "./src/core/graph/BasicBlock";

function guessLinesFromLabel(labelLines: string[], fileCode: string): [number, number] {
    // 把整个源码按行切分
    const lines = fileCode.split('\n');
    let startLineIndex = -1;
    let endLineIndex = -1;
  
    if (labelLines.length > 0) {
      // 1. 找第一行
      const firstLineSnippet = labelLines[0].trim();
      for (let i = 0; i < lines.length; i++) {
        // 用 includes()、startsWith() 或别的匹配策略
        if (lines[i].includes(firstLineSnippet)) {
          startLineIndex = i;
          break;
        }
      }
  
      // 2. 找最后一行
      const lastLineSnippet = labelLines[labelLines.length - 1].trim();
      for (let i = lines.length - 1; i >= 0; i--) {
        if (lines[i].includes(lastLineSnippet)) {
          endLineIndex = i;
          break;
        }
      }
    }
  
    // 约定：如果没找到，就当做 0（或 1）
    if (startLineIndex === -1) startLineIndex = 0;
    if (endLineIndex === -1) endLineIndex = 0;
  
    // 最后转成 1-based 行号
    return [startLineIndex + 1, endLineIndex + 1];
  }
  
  class JsonCfgPrinter {
    
    /**
     * 将单个 ArkFile 转为 JSON 对象
     */
    public convertArkFileToJson(arkFile: ArkFile): any {
      const fileName = arkFile.getName();
      const fileCode = arkFile.getCode() ?? "";
  
      let startLine = Number.MAX_SAFE_INTEGER;
      let endLine = -1;
  
      const fileJson: any = {
        type: "file",
        name: fileName,
        label: fileName,
        start_line: 0,
        end_line: 0,
        classes: [],
        functions: [],
        blocks: [],
        simplified_code: fileCode,
      };
  
      // 遍历 ArkClass
      for (const arkClass of arkFile.getClasses()) {
        const clsJson = this.convertArkClassToJson(arkClass, fileCode);
        fileJson.classes.push(clsJson);
  
        if (clsJson.start_line < startLine) {
          startLine = clsJson.start_line;
        }
        if (clsJson.end_line > endLine) {
          endLine = clsJson.end_line;
        }
      }
  
      // 行号修正
      if (startLine === Number.MAX_SAFE_INTEGER) {
        startLine = 1;
      }
      if (endLine < 1) {
        endLine = 1;
      }
      fileJson.start_line = startLine;
      fileJson.end_line = endLine;
  
      return fileJson;
    }
  
    /**
     * 将 ArkClass 转为 JSON
     */
    private convertArkClassToJson(arkClass: ArkClass, fileCode: string): any {
      let clsStart = arkClass.getLine() ?? 0;
      let clsEnd = clsStart;
  
      const classJson: any = {
        type: "class",
        name: arkClass.getName(),
        label: arkClass.getName(),
        start_line: clsStart,
        end_line: clsEnd,
        functions: [],
        classes: [],
        blocks: [],
        simplified_code: arkClass.getCode() ?? ""
      };
  
      for (const method of arkClass.getMethods()) {
        const methodJson = this.convertArkMethodToJson(method, fileCode);
        classJson.functions.push(methodJson);
  
        if (methodJson.start_line < clsStart) {
          clsStart = methodJson.start_line;
        }
        if (methodJson.end_line > clsEnd) {
          clsEnd = methodJson.end_line;
        }
      }
  
      classJson.start_line = clsStart;
      classJson.end_line = clsEnd;
      return classJson;
    }
  
    /**
     * 将 ArkMethod 转为 JSON
     */
    private convertArkMethodToJson(method: ArkMethod, fileCode: string): any {
      let mStart = method.getLine() ?? 0;
      let mEnd = mStart;
  
      const methodJson: any = {
        type: "function",
        name: method.getName(),
        label: method.getName(),
        start_line: mStart,
        end_line: mEnd,
        blocks: [],
        functions: [],
        classes: [],
        simplified_code: method.getCode() ?? ""
      };
  
      const body = method.getBody();
      if (body) {
        const cfg = body.getCfg();
        if (cfg) {
          const blocks = Array.from(cfg.getBlocks());
          const visited = new Set<BasicBlock>();
  
          // 逐个 block 转成 JSON
          for (const block of blocks) {
            const blockJson = this.convertBlockRecursive(block, fileCode, visited);
            methodJson.blocks.push(blockJson);
  
            // 更新方法的 min/max
            if (blockJson.start_line < mStart) {
              mStart = blockJson.start_line;
            }
            if (blockJson.end_line > mEnd) {
              mEnd = blockJson.end_line;
            }
          }
        }
      }
  
      methodJson.start_line = mStart;
      methodJson.end_line = mEnd;
      return methodJson;
    }
  
    /**
     * 将一个 BasicBlock 递归地转为 JSON，并从 label 的文本推断它的大致行号。
     */
    private convertBlockRecursive(block: BasicBlock, fileCode: string, visited: Set<BasicBlock>): any {
      if (visited.has(block)) {
        return {
          type: "blockRef",
          label: "(RepeatedBlock)"
        };
      }
      visited.add(block);
  
      // 收集 block 内的语句文本
      const stmts: Stmt[] = block.getStmts();
      const labelLines: string[] = [];
  
      for (const stmt of stmts) {
        labelLines.push(stmt.toString());
      }
  
      // 计算 block 的 label
      const labelText = labelLines.join("\n");
  
      // 用我们的辅助函数“猜测”行号
      const labelSplit = labelLines;  // 也就是 labelLines
      let [startLine, endLine] = guessLinesFromLabel(labelSplit, fileCode);
  
      const blockJson: any = {
        type: "block",
        name: "Block?",
        label: labelText,
        start_line: startLine,
        end_line: endLine,
        successors: []
      };
  
      // 处理后继
      for (const succBlock of block.getSuccessors()) {
        const succJson = this.convertBlockRecursive(succBlock, fileCode, visited);
        blockJson.successors.push(succJson);
      }
  
      return blockJson;
    }
  }
  

/************************************************************
 * 下面就是“主流程”逻辑：
 * 1. 构建Scene
 * 2. 调用PrinterBuilder输出.dot
 * 3. 使用JsonCfgPrinter对每个ArkFile生成JSON并分别输出到文件
 ************************************************************/
(function main() {
    let config: SceneConfig = new SceneConfig();
    let projectName = 'ts_files';
  
    // 构建场景配置
    config.buildFromProjectDir(path.join(__dirname, 'tests/projects', projectName));
  
    // 创建 Scene
    let scene = new Scene();
    scene.buildSceneFromProjectDir(config);
    scene.inferTypes();
  
    // 原 PrinterBuilder 输出 .dot
    // const printerBuilder = new PrinterBuilder();
    // for (const arkFile of scene.getFiles()) {
    //   const dotPath = `out/${projectName}/${arkFile.getName()}.dot`;
    //   printerBuilder.dumpToDot(arkFile, dotPath);
    // }
  
    // 用 JsonCfgPrinter，每个文件输出 JSON
    const jsonPrinter = new JsonCfgPrinter();
    for (const arkFile of scene.getFiles()) {
      const fileJson = jsonPrinter.convertArkFileToJson(arkFile);
      const outPath = `ts_cfg/${arkFile.getName()}.json`;
      fs.writeFileSync(outPath, JSON.stringify(fileJson, null, 2), "utf-8");
      console.log(`JSON exported: ${outPath}`);
    }
  })();