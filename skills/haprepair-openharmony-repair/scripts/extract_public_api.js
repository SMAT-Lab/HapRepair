#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const typescriptPath = process.env.HAPREPAIR_TYPESCRIPT ||
  '/home/zhihao/deveco/command-line-tools/codelinter/linter/arkPerfCheck/node_modules/arkanalyzer/node_modules/ohos-typescript'
const ts = require(typescriptPath)

const root = path.resolve(process.argv[2])
const ignored = new Set(['.git', '.haprepair', '.exp_agent', '.hvigor', '.idea', 'build', 'node_modules', 'oh_modules'])

function walk(dir, output = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ignored.has(entry.name)) continue
    const target = path.join(dir, entry.name)
    if (entry.isDirectory()) walk(target, output)
    else output.push(target)
  }
  return output
}

function normalize(value) {
  return value.replace(/\s+/g, ' ').trim()
}

function resolveModule(fromFile, specifier) {
  if (!specifier.startsWith('.')) return null
  const base = path.resolve(path.dirname(fromFile), specifier)
  for (const candidate of [base, `${base}.ets`, `${base}.ts`, path.join(base, 'Index.ets'), path.join(base, 'index.ets')]) {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate
  }
  return null
}

const sourceCache = new Map()
function sourceFile(file) {
  if (!sourceCache.has(file)) {
    sourceCache.set(file, ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS))
  }
  return sourceCache.get(file)
}

function hasModifier(node, kind) {
  return Boolean(node.modifiers && node.modifiers.some(item => item.kind === kind))
}

function importsFor(file) {
  const imports = new Map()
  for (const statement of sourceFile(file).statements) {
    if (!ts.isImportDeclaration(statement) || !statement.importClause) continue
    const target = resolveModule(file, statement.moduleSpecifier.text)
    if (!target) continue
    const clause = statement.importClause
    if (clause.name) imports.set(clause.name.text, { file: target, symbol: 'default' })
    const bindings = clause.namedBindings
    if (bindings && ts.isNamedImports(bindings)) {
      for (const element of bindings.elements) {
        imports.set(element.name.text, { file: target, symbol: element.propertyName ? element.propertyName.text : element.name.text })
      }
    }
  }
  return imports
}

function memberSignature(member, sf) {
  if (hasModifier(member, ts.SyntaxKind.PrivateKeyword) || hasModifier(member, ts.SyntaxKind.ProtectedKeyword)) return null
  const modifiers = (member.modifiers || []).map(item => normalize(item.getText(sf))).filter(item => !item.startsWith('@')).join(' ')
  const name = member.name ? normalize(member.name.getText(sf)) : 'constructor'
  const optional = member.questionToken ? '?' : ''
  const typeParameters = member.typeParameters ? `<${member.typeParameters.map(item => normalize(item.getText(sf))).join(',')}>` : ''
  const parameters = member.parameters ? `(${member.parameters.map(item => normalize(item.getText(sf))).join(',')})` : ''
  const type = member.type ? `:${normalize(member.type.getText(sf))}` : ''
  const prefix = modifiers ? `${modifiers} ` : ''
  return `${ts.SyntaxKind[member.kind]} ${prefix}${name}${optional}${typeParameters}${parameters}${type}`
}

function declarationFingerprint(file, symbol, seen) {
  const key = `${file}::${symbol}`
  if (seen.has(key)) return { kind: 'cycle', symbol }
  seen.add(key)
  const sf = sourceFile(file)
  const imports = importsFor(file)
  if (imports.has(symbol)) {
    const target = imports.get(symbol)
    return declarationFingerprint(target.file, target.symbol, seen)
  }
  for (const statement of sf.statements) {
    const name = statement.name && statement.name.text
    if (name !== symbol && !(symbol === 'default' && hasModifier(statement, ts.SyntaxKind.DefaultKeyword))) continue
    if (ts.isClassDeclaration(statement) || ts.isInterfaceDeclaration(statement) || ts.isStructDeclaration && ts.isStructDeclaration(statement)) {
      return {
        kind: ts.SyntaxKind[statement.kind],
        type_parameters: statement.typeParameters ? statement.typeParameters.map(item => normalize(item.getText(sf))) : [],
        heritage: statement.heritageClauses ? statement.heritageClauses.map(item => normalize(item.getText(sf))) : [],
        members: statement.members.map(item => memberSignature(item, sf)).filter(Boolean).sort(),
      }
    }
    if (ts.isFunctionDeclaration(statement)) {
      return { kind: 'FunctionDeclaration', signature: memberSignature(statement, sf) }
    }
    if (ts.isTypeAliasDeclaration(statement)) return { kind: 'TypeAliasDeclaration', type: normalize(statement.type.getText(sf)) }
    if (ts.isEnumDeclaration(statement)) return { kind: 'EnumDeclaration', members: statement.members.map(item => normalize(item.getText(sf))) }
  }
  for (const statement of sf.statements) {
    if (!ts.isVariableStatement(statement)) continue
    for (const declaration of statement.declarationList.declarations) {
      if (declaration.name.getText(sf) === symbol) return { kind: 'VariableDeclaration', type: declaration.type ? normalize(declaration.type.getText(sf)) : null }
    }
  }
  return exportedSymbol(file, symbol, seen)
}

function exportedSymbol(file, exportedName, seen = new Set()) {
  const sf = sourceFile(file)
  const imports = importsFor(file)
  for (const statement of sf.statements) {
    if (ts.isExportDeclaration(statement) && statement.exportClause && ts.isNamedExports(statement.exportClause)) {
      for (const element of statement.exportClause.elements) {
        if (element.name.text !== exportedName) continue
        const local = element.propertyName ? element.propertyName.text : element.name.text
        if (statement.moduleSpecifier) {
          const target = resolveModule(file, statement.moduleSpecifier.text)
          return target ? declarationFingerprint(target, local, seen) : { kind: 'unresolved', symbol: local }
        }
        if (imports.has(local)) {
          const target = imports.get(local)
          return declarationFingerprint(target.file, target.symbol, seen)
        }
        return declarationFingerprint(file, local, seen)
      }
    }
    if (statement.name && statement.name.text === exportedName && hasModifier(statement, ts.SyntaxKind.ExportKeyword)) {
      return declarationFingerprint(file, exportedName, seen)
    }
  }
  return { kind: 'unresolved', symbol: exportedName }
}

function exportedNames(file) {
  const names = []
  for (const statement of sourceFile(file).statements) {
    if (ts.isExportDeclaration(statement) && statement.exportClause && ts.isNamedExports(statement.exportClause)) {
      for (const element of statement.exportClause.elements) names.push(element.name.text)
    } else if (statement.name && hasModifier(statement, ts.SyntaxKind.ExportKeyword)) {
      names.push(statement.name.text)
    }
  }
  return [...new Set(names)].sort()
}

const entrypoints = []
for (const config of walk(root).filter(file => path.basename(file) === 'oh-package.json5')) {
  const match = fs.readFileSync(config, 'utf8').match(/["']main["']\s*:\s*["']([^"']+)["']/)
  if (!match) continue
  const entry = path.resolve(path.dirname(config), match[1])
  if (fs.existsSync(entry) && /\.(ets|ts)$/.test(entry)) entrypoints.push(entry)
}

const api = {}
for (const entry of [...new Set(entrypoints)].sort()) {
  const relative = path.relative(root, entry).split(path.sep).join('/')
  for (const name of exportedNames(entry)) api[`${relative}::${name}`] = exportedSymbol(entry, name)
}
process.stdout.write(JSON.stringify({ schema_version: 1, entrypoints: entrypoints.map(file => path.relative(root, file).split(path.sep).join('/')).sort(), api }, null, 2) + '\n')
