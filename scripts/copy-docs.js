import fs from 'fs'
import path from 'path'

const src = path.resolve('knowledge/build')
const dest = path.resolve('dist/knowledge')

if (!fs.existsSync(src)) {
  console.error('[copy-docs] knowledge/build 不存在，请先运行 pnpm docs:build')
  process.exit(1)
}

fs.rmSync(dest, { recursive: true, force: true })
fs.cpSync(src, dest, { recursive: true })
console.log('[copy-docs] knowledge/build -> dist/knowledge 完成')
