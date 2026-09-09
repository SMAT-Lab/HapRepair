#!/usr/bin/env node

const ts = require('/home/zhihao/deveco/command-line-tools/codelinter/linter/arkPerfCheck/node_modules/arkanalyzer/node_modules/ohos-typescript')

let source = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', chunk => { source += chunk })
process.stdin.on('end', () => {
  const file = process.argv[2] || 'candidate.ets'
  const parsed = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const diagnostics = parsed.parseDiagnostics.map(item => ({
    start: item.start,
    length: item.length,
    message: ts.flattenDiagnosticMessageText(item.messageText, ' '),
  }))
  process.stdout.write(JSON.stringify({ valid: diagnostics.length === 0, diagnostics }) + '\n')
})
