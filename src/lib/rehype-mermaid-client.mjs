import { visit } from 'unist-util-visit';

/**
 * Rehype plugin that marks mermaid code blocks for client-side rendering
 */
export default function rehypeMermaidClient() {
    return (tree) => {
        visit(tree, 'element', (node) => {
            // Find pre > code.language-mermaid elements
            if (
                node.tagName === 'pre' &&
                node.children &&
                node.children.length > 0
            ) {
                const codeNode = node.children[0];

                // Check if it's a code element with mermaid language class
                if (
                    codeNode?.tagName === 'code' &&
                    codeNode?.properties?.className
                ) {
                    const classes = Array.isArray(codeNode.properties.className)
                        ? codeNode.properties.className
                        : [codeNode.properties.className];

                    const isMermaid = classes.some(cls =>
                        cls === 'language-mermaid' || cls.includes('mermaid')
                    );

                    if (isMermaid) {
                        // Extract the mermaid diagram text
                        const diagramText = codeNode.children
                            .map(child => {
                                if (child.type === 'text') {
                                    return child.value;
                                } else if (child.type === 'element') {
                                    // Handle nested elements (e.g., from syntax highlighting)
                                    return child.children?.map(c => c.value || '').join('') || '';
                                }
                                return '';
                            })
                            .join('')
                            .trim();

                        // Replace the pre element with a div that will be processed by mermaid on the client
                        node.tagName = 'div';
                        node.properties = {
                            className: ['mermaid-diagram'],
                            'data-diagram': diagramText,
                        };
                        node.children = [{
                            type: 'text',
                            value: diagramText,
                        }];
                    }
                }
            }
        });
    };
}

