import { input } from '@inquirer/prompts'
import fs from 'fs'
import path from 'path'
import { isFileNameSafe } from './utils.js'

function getGalleryFullPath(fileName) {
  return path.join('./src/content/gallery', `${fileName}.md`)
}

const fileName = await input({
  message: '请输入文件名称',
  validate: (value) => {
    if (!isFileNameSafe(value)) {
      return '文件名只能包含字母、数字和连字符'
    }
    const fullPath = getGalleryFullPath(value)
    if (fs.existsSync(fullPath)) {
      return `${fullPath} 已存在`
    }
    return true
  },
})

const title = await input({
  message: '请输入相册标题',
})

const album = await input({
  message: '请输入所属相册名称',
})

const content = `---
title: ${title}
album: ${album}
images: []
videos: []
date: ${new Date().toISOString()}
draft: false
comments: true
---
`

const fullPath = getGalleryFullPath(fileName)
fs.writeFileSync(fullPath, content)
console.log(`${fullPath} 创建成功`)
