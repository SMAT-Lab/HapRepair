/***************************************************
 * analyze_ts_files.ts
 *
 * 功能：
 *  1) 遍历指定目录及子目录下所有 .ts 文件
 *  2) 逐一读取其内容，用正则分词 (去掉 \s+ 以免捕获空格)
 *  3) 用 TypeScript 创建 AST
 *  4) 特殊处理 SourceFile，让其从 [0, code.length) 覆盖整份文件
 *  5) 给每个 AST 节点标注 start_token / end_token
 *  6) 最终只输出 AST JSON (跳过 tokens 列表)
 ****************************************************/

import * as fs from 'fs';
import * as path from 'path';
import * as ts from 'typescript';

/**
 * 1. 模拟 Python 的 splitlines(keepends=True)，
 *    将字符串拆分为行，并保留行末的 \n 或 \r\n。
 */
function splitLinesKeepEnds(code: string): string[] {
  const lines: string[] = [];
  const regex = /[^\r\n]*[\r\n]?/g; // 每次匹配一行(含行尾换行符)
  let match;
  while ((match = regex.exec(code)) !== null) {
    if (match[0].length === 0) break; // 到达末尾
    lines.push(match[0]);
  }
  return lines;
}

/**
 * 2. 使用正则分词，并计算行号。
 *    返回格式：Array<[startOffset, endOffset, tokenText, lineNumber]>
 *
 *    去掉了 '\\s+'，不再捕获普通空格
 *    保留 '\\n|\\r|\\t' (如也不想捕获换行/回车/Tab，可删除)
 */
function tokenizeCodeWithLines(code: string): Array<[number, number, string, number]> {
  const tokenPattern = new RegExp(
    [
      // 标识符
      '[A-Za-z_]\\w*',
      // 数字(包含小数)
      '[0-9]+(?:\\.[0-9]+)?',
      // 各种字符串
      '"[^"]*"',
      "'[^']*'",
      '`[^`]*`',
      // 单行注释 //...
      '//.*?(?=\\n|$)',
      // 多行注释 /* ... */
      '/\\*[\\s\\S]*?\\*/',
      // 箭头函数 =>
      '=>',
      // 相等操作符
      '===|!==|==|!=',
      // 逻辑操作符
      '&&|\\|\\|',
      // 其他操作符或分隔符
      '[-+*/=<>!&|^~?:;,.(){}[\\]]',
      // 换行/回车/制表符
      '\\n|\\r|\\t',
    ].join('|'),
    'gms' // 全局 + 多行 + dotAll
  );

  const lines = splitLinesKeepEnds(code);
  let currentLine = 1;
  let currentPos = 0; // 已累计的字符数(包含行尾换行)
  const tokensWithOffset: Array<[number, number, string, number]> = [];

  let match;
  while ((match = tokenPattern.exec(code)) !== null) {
    const tk = match[0];
    const startOffset = match.index;
    const endOffset = startOffset + tk.length;

    // 计算当前 token 的行号
    while (
      currentLine <= lines.length &&
      currentPos + lines[currentLine - 1].length <= startOffset
    ) {
      currentPos += lines[currentLine - 1].length;
      currentLine++;
    }

    // 若 token 全是空白(比如纯回车符?), 则跳过
    if (tk.trim().length === 0) {
      continue;
    }

    tokensWithOffset.push([startOffset, endOffset, tk, currentLine]);
  }
  return tokensWithOffset;
}

/**
 * 3. 将 TS AST 节点转换成 JSON，并加上 start_token / end_token
 *    (这些索引基于正则 tokens 数组)
 *
 *    关键改动：对 SourceFile 节点做特殊处理：
 *      => 范围直接设为 [0, code.length) 以覆盖全文件
 */
function nodeToJsonWithTokenIndex(
  node: ts.Node,
  sourceFile: ts.SourceFile,
  tokens: Array<[number, number, string, number]>,
  code: string // 传入完整源码，以便获取其长度
): any {
  // 处理子节点
  const childrenJson: any[] = [];
  node.forEachChild((child) => {
    childrenJson.push(nodeToJsonWithTokenIndex(child, sourceFile, tokens, code));
  });

  // 根据节点类型，决定它的区间
  let nodeStart: number;
  let nodeEnd: number;

  // 如果是最外层 SourceFile，就让它覆盖整个文件
  if (node.kind === ts.SyntaxKind.SourceFile) {
    nodeStart = 0;
    nodeEnd = code.length;
  } else {
    // 普通节点，依旧用 getStart() / getEnd()
    nodeStart = node.getStart(sourceFile);
    nodeEnd = node.getEnd();
  }

  // 找出落在 [nodeStart, nodeEnd) 区间内的 token
  const coveredTokens = tokens.filter((tk) => {
    const [tkStart, tkEnd] = tk;
    // 与区间 [nodeStart, nodeEnd) 有重叠即可
    return !(tkEnd <= nodeStart || tkStart >= nodeEnd);
  });

  let startTokenIndex = -1;
  let endTokenIndex = -1;
  if (coveredTokens.length > 0) {
    // 按出现顺序找到最小 / 最大 index
    // tokens 本身是按扫描顺序排列的，所以可以借用 indexOf
    const coveredIndices = coveredTokens.map((tk) => tokens.indexOf(tk));
    coveredIndices.sort((a, b) => a - b);
    startTokenIndex = coveredIndices[0];
    endTokenIndex = coveredIndices[coveredIndices.length - 1];
  }

  return {
    type: ts.SyntaxKind[node.kind],
    label: node.getText(sourceFile),
    start_token: startTokenIndex,
    end_token: endTokenIndex,
    children: childrenJson,
  };
}

/**
 * 4. 解析指定 .ts 文件成 AST，并给节点加上 tokenIndex
 *    最终只输出 AST，不输出完整 tokens
 */
function parseAndSaveAst(filePath: string, outDir: string): void {
  const code = fs.readFileSync(filePath, 'utf-8');

  // (a) 先用正则分词(跳过空格)
  const tokens = tokenizeCodeWithLines(code);

  // (b) 用 TS 解析 AST
  const sourceFile = ts.createSourceFile(
    filePath,
    code,
    ts.ScriptTarget.Latest,
    true // setParentNodes = true，便于调试
  );

  // (c) 把 AST 转成 JSON，最外层 SourceFile 强制覆盖 [0, code.length)
  const astJson = nodeToJsonWithTokenIndex(sourceFile, sourceFile, tokens, code);

  // (d) 写文件，只输出 AST
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  const outPath = path.join(outDir, path.basename(filePath, '.ts') + '.json');
  fs.writeFileSync(outPath, JSON.stringify(astJson, null, 2), 'utf-8');
  console.log(`[OK] AST JSON -> ${outPath}`);
}

/**
 * 5. 获取指定目录(含子目录)下的所有 .ts 文件
 */
function getAllTsFiles(dirPath: string): string[] {
  let results: string[] = [];
  const items = fs.readdirSync(dirPath, { withFileTypes: true });
  for (const item of items) {
    const itemPath = path.join(dirPath, item.name);
    if (item.isDirectory()) {
      results = results.concat(getAllTsFiles(itemPath));
    } else if (itemPath.endsWith('.ts')) {
      results.push(itemPath);
    }
  }
  return results;
}

/**
 * 6. 主函数：遍历并对每个 .ts 文件分词 + 解析 AST + 保存
 */
function main() {
  // 1. 指定要解析的目录
  //    比如 /home/arkanalyzer/tests/projects/ts_files
  const rootDir = path.join(__dirname, 'tests/projects/ts_files');

  // 2. 指定 AST JSON 输出目录
  const outDir = path.join(__dirname, 'ts_ast');

  // 3. 找出所有 .ts 文件
  const tsFiles = getAllTsFiles(rootDir);
  if (tsFiles.length === 0) {
    console.log(`[WARN] No .ts files found in ${rootDir}`);
    return;
  }
  console.log(`[INFO] Found .ts files:`, tsFiles);

  // 4. 依次处理
  for (const filePath of tsFiles) {
    parseAndSaveAst(filePath, outDir);
  }
}

main();
