import { defineConfig } from 'astro/config'
import { mkdir, writeFile } from 'node:fs/promises'
import { remarkReadingTime } from './src/plugins/remarkReadingTime'
import { rehypeMermaid } from './src/plugins/rehypeMermaid'
import { rehypeCodeBlock } from './src/plugins/rehypeCodeBlock'
import { rehypeTableBlock } from './src/plugins/rehypeTableBlock'
import { rehypeCodeHighlight } from './src/plugins/rehypeCodeHighlight'
import { rehypeImage } from './src/plugins/rehypeImage'
import { rehypeLink } from './src/plugins/rehypeLink'
import { rehypeHeading } from './src/plugins/rehypeHeading'
import remarkDirective from 'remark-directive'
import { remarkSpoiler } from './src/plugins/remarkSpoiler'
import { remarkEmbed } from './src/plugins/remarkEmbed'
import tailwind from '@astrojs/tailwind'
import react from '@astrojs/react'
import sitemap from '@astrojs/sitemap'
import { rehypeHeadingIds } from '@astrojs/markdown-remark'
import { site } from './src/config.json'
import { legacyPaths } from './src/legacy-redirects.mjs'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import swup from '@swup/astro'

const encodePath = (path) => new URL(path, site.url).pathname
const redirects = Object.fromEntries(
  Object.entries(legacyPaths).map(([from, to]) => [from, { destination: `${to}/`, status: 301 }]),
)

// https://astro.build/config
export default defineConfig({
  site: site.url,
  redirects,
  build: { redirects: false },
  integrations: [
    tailwind(),
    react(),
    sitemap({
      filter: (page) =>
        !Object.hasOwn(legacyPaths, decodeURIComponent(new URL(page).pathname).replace(/\/$/, '')),
    }),
    {
      name: 'legacy-redirects',
      hooks: {
        'astro:build:done': async ({ dir }) => {
          // Local previews and static hosts still need pages that preserve incoming anchors.
          await Promise.all(
            Object.entries(redirects).map(async ([from, { destination }]) => {
              const target = encodePath(destination)
              const folder = new URL(`.${encodePath(from)}/`, dir)
              await mkdir(folder, { recursive: true })
              await writeFile(
                new URL('index.html', folder),
                `<!doctype html>
<html lang="${site.lang}">
<head>
  <meta charset="utf-8">
  <meta name="robots" content="noindex">
  <link rel="canonical" href="${new URL(target, site.url)}">
  <title>页面已迁移</title>
  <script>location.replace(${JSON.stringify(target)} + location.search + location.hash)</script>
  <noscript><meta http-equiv="refresh" content="0;url=${target}"></noscript>
</head>
<body><a href="${target}">前往新页面</a></body>
</html>
`,
              )
            }),
          )
        },
      },
    },
    swup({
      theme: false,
      animationClass: 'swup-transition-',
      containers: ['main'],
      morph: ['[component-export="Provider"]'],
    }),
  ],
  markdown: {
    syntaxHighlight: false,
    smartypants: false,
    remarkPlugins: [remarkMath, remarkDirective, remarkEmbed, remarkSpoiler, remarkReadingTime],
    rehypePlugins: [
      rehypeHeadingIds,
      rehypeKatex,
      rehypeLink,
      rehypeImage,
      rehypeHeading,
      rehypeMermaid,
      rehypeCodeBlock,
      rehypeCodeHighlight,
      rehypeTableBlock,
    ],
    remarkRehype: { footnoteLabel: '参考', footnoteBackLabel: '返回正文' },
  },
  vite: {
    build: {
      rollupOptions: {
        external: ['/pagefind/pagefind.js'],
      },
    },
  },
})
