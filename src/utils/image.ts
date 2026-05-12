/**
 * 图片处理工具函数
 * 支持缩略图生成和模糊占位符
 */

/**
 * 生成缩略图 URL
 * 支持常见的图片 CDN 参数格式（Cloudinary, imgix, Unsplash 等）
 */
export function getThumbnailUrl(
  imageUrl: string,
  width: number = 400,
  height: number = 300,
  quality: number = 80,
): string {
  try {
    const url = new URL(imageUrl)

    // Unsplash 图片处理
    if (url.hostname.includes('unsplash.com')) {
      url.searchParams.set('w', String(width))
      url.searchParams.set('q', String(quality))
      url.searchParams.set('fit', 'crop')
      return url.toString()
    }

    // Cloudinary 图片处理
    if (url.hostname.includes('cloudinary.com')) {
      // Cloudinary 使用路径参数: /upload/w_400,h_300,q_80,c_fill/...
      const pathMatch = url.pathname.match(/^(.*\/upload\/)(.*)$/)
      if (pathMatch) {
        const basePath = pathMatch[1]
        const imagePath = pathMatch[2]
        const transformations = `w_${width},h_${height},q_${quality},c_fill`
        url.pathname = `${basePath}${transformations}/${imagePath}`
      }
      return url.toString()
    }

    // imgix 图片处理
    if (url.hostname.includes('imgix')) {
      url.searchParams.set('w', String(width))
      url.searchParams.set('h', String(height))
      url.searchParams.set('q', String(quality))
      url.searchParams.set('fit', 'crop')
      return url.toString()
    }

    // 默认：追加宽度参数（某些 CDN 支持）
    url.searchParams.set('w', String(width))
    url.searchParams.set('h', String(height))
    url.searchParams.set('q', String(quality))

    return url.toString()
  } catch {
    // 如果 URL 解析失败，返回原始 URL
    return imageUrl
  }
}
