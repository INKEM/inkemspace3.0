import { getCollection } from 'astro:content'

// 获取所有文章
async function getAllPosts() {
  const allPosts = await getCollection('posts', ({ data }) => {
    return import.meta.env.PROD ? data.draft !== true : true
  })

  return allPosts
}

// 获取所有文章，发布日期升序
async function getChronologicalPosts() {
  const allPosts = await getAllPosts()

  return allPosts.sort((a, b) => {
    return a.data.date.valueOf() - b.data.date.valueOf()
  })
}

// 获取所有文章，发布日期降序
export async function getNewestPosts() {
  const allPosts = await getAllPosts()

  return allPosts.sort((a, b) => {
    return b.data.date.valueOf() - a.data.date.valueOf()
  })
}

// 获取所有文章，置顶优先，发布日期降序
export async function getSortedPosts() {
  const allPosts = await getAllPosts()

  return allPosts.sort((a, b) => {
    if (a.data.sticky !== b.data.sticky) {
      return b.data.sticky - a.data.sticky
    } else {
      return b.data.date.valueOf() - a.data.date.valueOf()
    }
  })
}

// 获取分页文章
export async function getSortedPostsPaginated(page: number, perPage: number) {
  const allPosts = await getSortedPosts()
  const totalPages = Math.ceil(allPosts.length / perPage)
  const startIdx = (page - 1) * perPage
  const endIdx = startIdx + perPage

  return {
    data: allPosts.slice(startIdx, endIdx),
    totalPages,
    currentPage: page,
    total: allPosts.length,
  }
}

// 获取所有文章的字数
export async function getAllPostsWordCount() {
  const allPosts = await getAllPosts()

  return allPosts.reduce((count, post) => {
    const text = post.body
      .replace(/[#*_`\[\](){}>\-!|~]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
    return count + (text ? text.split(' ').length : 0)
  }, 0)
}

// 转换为 URL 安全的 slug，删除点，空格转为短横线，大写转为小写
export function slugify(text: string) {
  return text.replace(/\./g, '').replace(/\s/g, '-').toLowerCase()
}

// 获取所有分类
export async function getAllCategories() {
  const newestPosts = await getChronologicalPosts()

  const allCategories = newestPosts.reduce<{ slug: string; name: string; count: number }[]>(
    (acc, cur) => {
      if (cur.data.category) {
        const slug = slugify(cur.data.category)
        const index = acc.findIndex((category) => category.slug === slug)
        if (index === -1) {
          acc.push({
            slug,
            name: cur.data.category,
            count: 1,
          })
        } else {
          acc[index].count += 1
        }
      }
      return acc
    },
    [],
  )

  return allCategories
}

// 获取所有标签
export async function getAllTags() {
  const newestPosts = await getChronologicalPosts()

  const allTags = newestPosts.reduce<{ slug: string; name: string; count: number }[]>(
    (acc, cur) => {
      cur.data.tags.forEach((tag) => {
        const slug = slugify(tag)
        const index = acc.findIndex((tag) => tag.slug === slug)
        if (index === -1) {
          acc.push({
            slug,
            name: tag,
            count: 1,
          })
        } else {
          acc[index].count += 1
        }
      })
      return acc
    },
    [],
  )

  return allTags
}

// 获取热门标签
export async function getHotTags(len = 5) {
  const allTags = await getAllTags()

  return allTags
    .sort((a, b) => {
      return b.count - a.count
    })
    .slice(0, len)
}

// ==================== Gallery Collection ====================

// 获取所有相册
async function getAllGallery() {
  const allGallery = await getCollection('gallery', ({ data }) => {
    return import.meta.env.PROD ? data.draft !== true : true
  })

  return allGallery
}

// 获取所有相册，发布日期降序
export async function getSortedGallery() {
  const allGallery = await getAllGallery()

  return allGallery.sort((a, b) => {
    return b.data.date.valueOf() - a.data.date.valueOf()
  })
}

// 获取分页相册（用于相册首页分页）
export async function getSortedGalleryPaginated(page: number, perPage: number) {
  const allGallery = await getSortedGallery()
  const totalPages = Math.ceil(allGallery.length / perPage)
  const startIdx = (page - 1) * perPage
  const endIdx = startIdx + perPage

  return {
    data: allGallery.slice(startIdx, endIdx),
    totalPages,
    currentPage: page,
    total: allGallery.length,
  }
}

// 获取单个相册条目
export async function getGalleryEntry(slug: string) {
  const allGallery = await getAllGallery()
  return allGallery.find((entry) => entry.slug === slug)
}

// 按相册分页获取图片
export async function getSortedGalleryByAlbumPaginated(
  albumSlug: string,
  page: number,
  perPage: number,
) {
  const allGallery = await getSortedGallery()
  const filtered = allGallery.filter(
    (entry) => slugify(entry.data.album) === slugify(albumSlug),
  )
  const totalPages = Math.ceil(filtered.length / perPage)
  const startIdx = (page - 1) * perPage
  const endIdx = startIdx + perPage

  return {
    data: filtered.slice(startIdx, endIdx),
    totalPages,
    currentPage: page,
    total: filtered.length,
  }
}

// 获取所有相册分类（album）
export async function getAllAlbums() {
  const allGallery = await getAllGallery()

  const allAlbums = allGallery.reduce<{ slug: string; name: string; count: number }[]>(
    (acc, cur) => {
      const slug = slugify(cur.data.album)
      const index = acc.findIndex((album) => album.slug === slug)
      if (index === -1) {
        acc.push({
          slug,
          name: cur.data.album,
          count: 1,
        })
      } else {
        acc[index].count += 1
      }
      return acc
    },
    [],
  )

  return allAlbums
}

// 按相册获取图片
export async function getGalleryByAlbum(album: string) {
  const allGallery = await getSortedGallery()

  return allGallery.filter((gallery) => slugify(gallery.data.album) === slugify(album))
}

// ==================== Notes Collection ====================

// 获取所有杂记
async function getAllNotes() {
  const allNotes = await getCollection('notes', ({ data }) => {
    return import.meta.env.PROD ? data.draft !== true : true
  })

  return allNotes
}

// 获取所有杂记，发布日期降序
export async function getSortedNotes() {
  const allNotes = await getAllNotes()

  return allNotes.sort((a, b) => {
    return b.data.date.valueOf() - a.data.date.valueOf()
  })
}

// 获取分页杂记
export async function getSortedNotesPaginated(page: number, perPage: number) {
  const allNotes = await getSortedNotes()
  const totalPages = Math.ceil(allNotes.length / perPage)
  const startIdx = (page - 1) * perPage
  const endIdx = startIdx + perPage

  return {
    data: allNotes.slice(startIdx, endIdx),
    totalPages,
    currentPage: page,
    total: allNotes.length,
  }
}

// 获取所有杂记标签
export async function getAllNoteTags() {
  const allNotes = await getAllNotes()

  const allTags = allNotes.reduce<{ slug: string; name: string; count: number }[]>(
    (acc, cur) => {
      cur.data.tags.forEach((tag) => {
        const slug = slugify(tag)
        const index = acc.findIndex((tag) => tag.slug === slug)
        if (index === -1) {
          acc.push({
            slug,
            name: tag,
            count: 1,
          })
        } else {
          acc[index].count += 1
        }
      })
      return acc
    },
    [],
  )

  return allTags
}

// 按标签获取杂记
export async function getNotesByTag(tag: string) {
  const allNotes = await getSortedNotes()

  return allNotes.filter((note) => note.data.tags.some((t) => slugify(t) === slugify(tag)))
}

// 文本截断工具函数
export function truncateText(text: string, length: number = 150): string {
  if (text.length <= length) return text
  return text.substring(0, length).trim() + '...'
}
