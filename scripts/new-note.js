import { input } from '@inquirer/prompts'
import fs from 'fs'
import path from 'path'
import { isFileNameSafe } from './utils.js'

function getNoteFullPath(fileName) {
  return path.join('./src/content/notes', `${fileName}.md`)
}

const fileName = await input({
  message: '请输入文件名称',
  validate: (value) => {
    if (!isFileNameSafe(value)) {
      return '文件名只能包含字母、数字和连字符'
    }
    const fullPath = getNoteFullPath(value)
    if (fs.existsSync(fullPath)) {
      return `${fullPath} 已存在`
    }
    return true
  },
})

const title = await input({
  message: '请输入杂记标题',
})

const content = `---
title: ${title}
date: ${new Date().toISOString()}
tags: []
draft: false
---
`

const fullPath = getNoteFullPath(fileName)
fs.writeFileSync(fullPath, content)
console.log(`${fullPath} 创建成功`)
