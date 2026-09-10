import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import type {ReactNode} from 'react';

// 与博客(astro-gyoza)的亮暗模式联动:
// 博客将用户选择保存在 localStorage['gyoza-theme'](light/dark/system),
// 这里在启动时把显式的 light/dark 同步给 Docusaurus 的 theme 状态;
// system 或未设置时保持 Docusaurus 默认的跟随系统行为。
// storage 事件只在同源的其他页签触发,博客页切换主题时知识库页会实时跟随。
function syncFromBlog() {
  const blogTheme = localStorage.getItem('gyoza-theme');
  if (blogTheme === 'dark' || blogTheme === 'light') {
    localStorage.setItem('theme', blogTheme);
    document.documentElement.setAttribute('data-theme', blogTheme);
  }
}

if (ExecutionEnvironment.canUseDOM) {
  syncFromBlog();
  window.addEventListener('storage', syncFromBlog);
}

export default function ThemeSync(): ReactNode {
  return null;
}
