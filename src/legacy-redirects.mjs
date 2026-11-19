import legacy from './legacy-redirects.json' with { type: 'json' }

export const legacyPaths = {
  ...legacy.paths,
  ...Object.fromEntries(
    Object.entries(legacy.essays).map(([slug, path]) => [path, `/notes/${slug}`]),
  ),
}
