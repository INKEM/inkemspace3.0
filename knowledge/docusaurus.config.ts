import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// 暗色代码主题基于 VS Code Dark+,背景色对齐博客 Shiki 的 github-dark
const githubDarkTheme = {
  ...prismThemes.vsDark,
  plain: {...prismThemes.vsDark.plain, backgroundColor: '#24292e'},
};

const config: Config = {
  title: '知识库',
  tagline: '一杯为品的知识库',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://inkem.space',
  // 知识库部署在博客域名下的 /knowledge/ 子路径
  baseUrl: '/knowledge/',

  organizationName: 'INKEM', // Usually your GitHub org/user name.
  projectName: 'inkemspace3.0', // Usually your repo name.

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  clientModules: ['./src/clientModules/themeSync.ts'],

  presets: [
    [
      'classic',
      {
        docs: {
          // 文档直接作为知识库首页,/knowledge/ 即文档入口
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          // 数学公式渲染,与博客的 remark-math + KaTeX 栈一致
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          // Remove this to remove the "edit this page" links.
          editUrl: 'https://github.com/INKEM/inkemspace3.0/edit/main/knowledge/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: '知识库',
      items: [
        {
          href: 'https://inkem.space',
          label: '博客',
          position: 'right',
        },
        {
          href: 'https://github.com/INKEM',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'light',
      copyright: `Copyright © ${new Date().getFullYear()} Inkem · 一杯为品`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: githubDarkTheme,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
