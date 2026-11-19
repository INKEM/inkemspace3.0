import { legacyPaths } from './src/legacy-redirects.mjs'

export const config = {
  redirects: [
    ...Object.entries(legacyPaths).flatMap(([from, to]) => {
      // Old knowledge URLs may arrive decoded or with Chinese text, spaces and ampersands encoded.
      const encoded = encodeURI(from)
      const spellings = new Set([from, encoded, encoded.replaceAll('&', '%26')])

      return [...spellings].map((path) => ({
        source: `${path.replace(/[\\:+*?{}()]/g, '\\$&')}{/}?`,
        destination: encodeURI(to),
        statusCode: 301,
      }))
    }),
    { source: '/knowledge/:path*{/}?', destination: '/archives', statusCode: 301 },
  ],
}
