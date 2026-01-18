import { visit } from 'unist-util-visit';

/**
 * Remark plugin that transforms mermaid code blocks for client-side rendering
 * This runs before syntax highlighting, so we can catch the raw code blocks
 */
export default function remarkMermaid() {
    return (tree) => {
        visit(tree, 'code', (node) => {
            // Check if this is a mermaid code block
            if (node.lang === 'mermaid') {
                // Transform the code block into a custom HTML block
                // that will bypass syntax highlighting
                node.type = 'html';
                node.value = `<div class="mermaid-diagram" data-diagram="${escapeHtml(node.value)}">${escapeHtml(node.value)}</div>`;
            }
        });
    };
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
