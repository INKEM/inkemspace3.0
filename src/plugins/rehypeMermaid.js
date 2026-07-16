import { h } from 'hastscript'
import { visit } from 'unist-util-visit'

export function rehypeMermaid() {
  return function (tree) {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'pre') return

      const codeEl = node.children[0]
      if (
        !codeEl ||
        codeEl.type !== 'element' ||
        codeEl.tagName !== 'code' ||
        !codeEl.properties
      ) {
        return
      }

      const classes = codeEl.properties.className
      if (
        !Array.isArray(classes) ||
        !classes.includes('language-mermaid')
      ) {
        return
      }

      const textNode = codeEl.children[0]
      const code =
        textNode && textNode.type === 'text' ? textNode.value : ''

      parent.children[index] = h('div', { class: 'mermaid' }, code)
    })
  }
}
