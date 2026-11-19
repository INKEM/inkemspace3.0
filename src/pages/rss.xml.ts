import type { APIContext } from 'astro'
import rss from '@astrojs/rss'
import { site } from '@/config.json'
import { getLegacyEssayPath, getNewestPosts, getNoteExcerpt, getSortedNotes } from '@/utils/content'

export async function GET(context: APIContext) {
  const [posts, notes] = await Promise.all([getNewestPosts(), getSortedNotes()])

  const items = [
    ...posts.map((post) => ({
      link: `/posts/${post.slug}`,
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.summary,
    })),
    ...notes.map((note) => {
      const oldPath = getLegacyEssayPath(note.slug)
      const oldGuid = oldPath && new URL(`${oldPath}/`, context.site).href
      return {
        link: `/notes/${note.slug}`,
        title: note.data.title,
        pubDate: note.data.date,
        description: getNoteExcerpt(note),
        customData: oldGuid ? `<guid isPermaLink="true">${oldGuid}</guid>` : undefined,
      }
    }),
  ].sort((a, b) => b.pubDate.valueOf() - a.pubDate.valueOf())

  return rss({
    title: site.title,
    description: site.description,
    site: context.site!,
    items,
    customData: `<language>${site.lang}</language>`,
  })
}
