'use strict'

// Hvigor exposes the ArkTS compiler's ignoreWarning switch only for widget
// builds. Inject the same compiler option for evaluator builds without
// modifying the project or the installed DevEco toolchain.
const fs = require('fs')
const Module = require('module')

const moduleSuffix = '/hvigor-ohos-plugin/src/tasks/ark-compile.js'

function shouldInject(filename) {
  return filename.replaceAll('\\', '/').endsWith(moduleSuffix)
}

function injectIgnoreWarning(source, filename) {
  const pattern = /const ([A-Za-z_$][\w$]*)=await this\.initDefaultArkCompileConfig\(\);/
  const match = source.match(pattern)
  if (!match) {
    throw new Error(`Unsupported Hvigor ArkCompile implementation: ${filename}`)
  }
  return source.replace(pattern, `${match[0]}${match[1]}.ignoreWarning=!0;`)
}

const originalLoader = Module._extensions['.js']

Module._extensions['.js'] = function loadWithArkTSDiagnosticPolicy(module, filename) {
  if (!shouldInject(filename)) {
    return originalLoader(module, filename)
  }

  const source = fs.readFileSync(filename, 'utf8')
  module._compile(injectIgnoreWarning(source, filename), filename)
}

module.exports = { injectIgnoreWarning, shouldInject }
