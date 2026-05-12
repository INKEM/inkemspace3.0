import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { getThumbnailUrl } from '@/utils/image'

interface Image {
  url: string
  alt?: string
  description?: string
}

interface Video {
  url: string
  cover?: string
  title?: string
}

interface Props {
  images?: Image[]
  videos?: Video[]
  title?: string
}

export default function GalleryGrid({
  images = [],
  videos = [],
  title = 'Gallery',
}: Props) {
  const loadedImagesRef = useRef<Set<string>>(new Set())
  const observerRef = useRef<IntersectionObserver | null>(null)
  const imgRefsMap = useRef<Map<string, HTMLImageElement>>(null!)

  if (!imgRefsMap.current) {
    imgRefsMap.current = new Map()
  }

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement
            const dataSrc = img.dataset.src

            if (dataSrc && !loadedImagesRef.current.has(dataSrc)) {
              img.src = dataSrc
              img.onload = () => {
                img.classList.add('loaded')
                loadedImagesRef.current.add(dataSrc)
              }
              observerRef.current?.unobserve(img)
            }
          }
        })
      },
      {
        rootMargin: '50px',
      },
    )

    document.querySelectorAll('[data-src]').forEach((img) => {
      observerRef.current?.observe(img)
    })

    return () => {
      observerRef.current?.disconnect()
    }
  }, [])

  const mediaItems = [
    ...images.map((img, idx) => ({
      id: `img-${idx}`,
      type: 'image' as const,
      src: img.url,
      alt: img.alt || `Gallery image ${idx + 1}`,
      description: img.description,
    })),
    ...videos.map((vid, idx) => ({
      id: `vid-${idx}`,
      type: 'video' as const,
      src: vid.url,
      cover: vid.cover,
      title: vid.title || `Video ${idx + 1}`,
    })),
  ]

  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null)
  const imageItems = mediaItems.filter((m) => m.type === 'image')

  const openLightbox = useCallback((index: number) => {
    setLightboxIndex(index)
    document.body.style.overflow = 'hidden'
  }, [])

  const closeLightbox = useCallback(() => {
    setLightboxIndex(null)
    document.body.style.overflow = ''
  }, [])

  const goToPrev = () => {
    if (imageItems.length === 0) return
    setLightboxIndex((prev) =>
      prev !== null ? (prev - 1 + imageItems.length) % imageItems.length : null,
    )
  }

  const goToNext = () => {
    if (imageItems.length === 0) return
    setLightboxIndex((prev) => (prev !== null ? (prev + 1) % imageItems.length : null))
  }

  useEffect(() => {
    if (lightboxIndex === null) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeLightbox()
      if (e.key === 'ArrowLeft') goToPrev()
      if (e.key === 'ArrowRight') goToNext()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [lightboxIndex, closeLightbox])

  return (
    <div className="gallery-grid-container">
      {title && <h2 className="text-2xl font-bold mb-6 text-primary">{title}</h2>}

      <div className="gallery-grid">
        {mediaItems.map((item) => {
          if (item.type === 'image') {
            const thumbnail = getThumbnailUrl(item.src, 400, 300)

            const imageIndex = imageItems.findIndex((m) => m.id === item.id)

            return (
              <div key={item.id} className="gallery-item image" onClick={() => openLightbox(imageIndex)}>
                <div className="image-wrapper">
                  <img
                    ref={(el) => {
                      if (el) imgRefsMap.current.set(item.id, el)
                    }}
                    data-src={item.src}
                    src={thumbnail}
                    alt={item.alt}
                    className="lazy-image cursor-zoom-in"
                    loading="lazy"
                    decoding="async"
                  />
                  <div className="overlay">
                    {item.description && <p className="description">{item.description}</p>}
                  </div>
                </div>
              </div>
            )
          } else {
            return (
              <div key={item.id} className="gallery-item video">
                <div className="video-wrapper">
                  {item.cover && (
                    <img
                      src={getThumbnailUrl(item.cover, 400, 300)}
                      alt={item.title}
                      className="video-cover"
                    />
                  )}
                  <div className="play-button">
                    <svg className="w-12 h-12" fill="white" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  </div>
                  <a
                    href={item.src}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="video-link"
                    title={item.title}
                  >
                    {item.title}
                  </a>
                </div>
              </div>
            )
          }
        })}
      </div>

      {mediaItems.length === 0 && (
        <div className="empty-state">
          <p className="text-gray-500">No media items to display</p>
        </div>
      )}

      {lightboxIndex !== null && imageItems[lightboxIndex] && createPortal(
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/90" onClick={closeLightbox}>
          <button
            className="absolute top-4 right-4 text-white text-3xl hover:opacity-70 z-10"
            onClick={closeLightbox}
          >
            ✕
          </button>

          {imageItems.length > 1 && (
            <>
              <button
                className="absolute left-4 top-1/2 -translate-y-1/2 text-white text-4xl hover:opacity-70 z-10"
                onClick={(e) => { e.stopPropagation(); goToPrev() }}
              >
                ‹
              </button>
              <button
                className="absolute right-4 top-1/2 -translate-y-1/2 text-white text-4xl hover:opacity-70 z-10"
                onClick={(e) => { e.stopPropagation(); goToNext() }}
              >
                ›
              </button>
            </>
          )}

          <div className="flex flex-col items-center max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <img
              src={imageItems[lightboxIndex].src}
              alt={imageItems[lightboxIndex].alt}
              className="max-w-full max-h-[85vh] object-contain"
            />
            {imageItems[lightboxIndex].description && (
              <p className="text-white/80 mt-4 text-sm">{imageItems[lightboxIndex].description}</p>
            )}
            {imageItems.length > 1 && (
              <p className="text-white/50 mt-2 text-xs">{lightboxIndex + 1} / {imageItems.length}</p>
            )}
          </div>
        </div>,
        document.body
      )}

      <style>{`
        .gallery-grid-container {
          width: 100%;
        }

        .gallery-grid {
          column-count: 3;
          column-gap: 1rem;
        }

        @media (max-width: 768px) {
          .gallery-grid {
            column-count: 2;
          }
        }

        @media (max-width: 480px) {
          .gallery-grid {
            column-count: 1;
          }
        }

        .gallery-item {
          break-inside: avoid;
          margin-bottom: 1rem;
          overflow: hidden;
          border-radius: 0.5rem;
          background: #f3f4f6;
          cursor: pointer;
        }

        .gallery-item.image {
          cursor: zoom-in;
        }

        .image-wrapper,
        .video-wrapper {
          width: 100%;
          position: relative;
          overflow: hidden;
        }

        .lazy-image,
        .video-cover {
          width: 100%;
          display: block;
          object-fit: cover;
          transition: transform 0.3s ease;
        }

        .gallery-item:hover .lazy-image,
        .gallery-item:hover .video-cover {
          transform: scale(1.05);
        }

        .overlay {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0);
          display: flex;
          align-items: flex-end;
          padding: 1rem;
          transition: background 0.3s ease;
        }

        .gallery-item:hover .overlay {
          background: rgba(0, 0, 0, 0.6);
        }

        .description {
          color: white;
          font-size: 0.875rem;
          line-height: 1.25rem;
          opacity: 0;
          transition: opacity 0.3s ease;
          max-height: 100%;
          overflow: hidden;
        }

        .gallery-item:hover .description {
          opacity: 1;
        }

        .play-button {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: rgba(255, 255, 255, 0.9);
          border-radius: 50%;
          padding: 0.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s ease;
          cursor: pointer;
        }

        .video-wrapper:hover .play-button {
          background: white;
          transform: translate(-50%, -50%) scale(1.1);
        }

        .video-link {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          text-decoration: none;
          color: inherit;
          cursor: pointer;
        }

        .empty-state {
          text-align: center;
          padding: 3rem 1rem;
          color: #9ca3af;
        }

        .lazy-image {
          opacity: 0.8;
          filter: blur(5px);
        }

        .lazy-image.loaded {
          opacity: 1;
          filter: blur(0);
          animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
      `}</style>
    </div>
  )
}
