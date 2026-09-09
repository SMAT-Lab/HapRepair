#!/usr/bin/env node

const fs = require('fs')
const path = require('path')

const typescriptPath = process.env.HAPREPAIR_TYPESCRIPT ||
  '/home/zhihao/deveco/command-line-tools/codelinter/linter/arkPerfCheck/node_modules/arkanalyzer/node_modules/ohos-typescript'
const ts = require(typescriptPath)
const root = path.resolve(process.argv[2])
const ignored = new Set(['.git', '.haprepair', '.hvigor', '.idea', 'build', 'node_modules', 'oh_modules'])

function walk(directory, output = []) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (ignored.has(entry.name)) continue
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) walk(target, output)
    else if (/\.(ets|ts)$/.test(entry.name)) output.push(target)
  }
  return output
}

function nodeName(node, sourceFile) {
  if (!node.name) return null
  return node.name.text || node.name.getText(sourceFile)
}

function inventory(file) {
  const sourceText = fs.readFileSync(file, 'utf8')
  const sourceFile = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const declarations = new Set()

  function visit(node, scope) {
    let childScope = scope
    const name = nodeName(node, sourceFile)
    const isStruct = Boolean(ts.isStructDeclaration && ts.isStructDeclaration(node))
    if (name && (ts.isClassDeclaration(node) || isStruct)) {
      childScope = scope ? `${scope}.${name}` : name
      declarations.add(`${isStruct ? 'struct' : 'class'}:${childScope}`)
    } else if (name && ts.isFunctionDeclaration(node)) {
      declarations.add(`function:${scope ? `${scope}.` : ''}${name}`)
    } else if (name && ts.isMethodDeclaration(node)) {
      declarations.add(`method:${scope ? `${scope}.` : ''}${name}`)
    }
    ts.forEachChild(node, child => visit(child, childScope))
  }

  visit(sourceFile, '')
  return [...declarations].sort()
}

const declarations = {}
for (const file of walk(root).sort()) {
  const relative = path.relative(root, file).split(path.sep).join('/')
  declarations[relative] = inventory(file)
}

process.stdout.write(JSON.stringify({ schema_version: 1, declarations }, null, 2) + '\n')
