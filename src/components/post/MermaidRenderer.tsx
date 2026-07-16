import { useEffect, useRef } from 'react'

export function MermaidRenderer() {
  const initializedRef = useRef(false)

  useEffect(() => {
    const blocks = document.querySelectorAll('.mermaid')
    if (blocks.length === 0) return

    function getBlocks(): HTMLElement[] {
      return Array.from(
        document.querySelectorAll<HTMLElement>('.mermaid'),
      )
    }

    function getCurrentTheme(): 'dark' | 'light' {
      return document.documentElement.getAttribute('data-theme') ===
        'dark'
        ? 'dark'
        : 'light'
    }

    async function render(theme?: 'dark' | 'light') {
      const themeToUse = theme || getCurrentTheme()
      const els = getBlocks()
      if (els.length === 0) return

      if (!initializedRef.current) {
        els.forEach((el) => {
          el.dataset.code = el.textContent || ''
        })
      } else {
        els.forEach((el) => {
          el.textContent = el.dataset.code || ''
        })
      }

      try {
        const { default: mermaid } = await import('mermaid')
        mermaid.initialize({
          startOnLoad: false,
          theme: themeToUse === 'dark' ? 'dark' : 'default',
        })
        await mermaid.run({
          querySelector: '.mermaid',
          suppressErrors: true,
        })
        initializedRef.current = true
      } catch (err) {
        console.error('[Mermaid] Render error:', err)
      }
    }

    render()

    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (
          m.type === 'attributes' &&
          m.attributeName === 'data-theme'
        ) {
          render(
            document.documentElement.getAttribute('data-theme') ===
              'dark'
              ? 'dark'
              : 'light',
          )
          break
        }
      }
    })
    observer.observe(document.documentElement, { attributes: true })

    function handleSwupReplace() {
      initializedRef.current = false
      setTimeout(() => render(), 100)
    }
    document.addEventListener('swup:content:replace', handleSwupReplace)

    return () => {
      observer.disconnect()
      document.removeEventListener(
        'swup:content:replace',
        handleSwupReplace,
      )
    }
  }, [])

  return null
}
