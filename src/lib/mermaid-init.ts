import mermaid from 'mermaid';

// Initialize mermaid with dark theme
mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    themeVariables: {
        fontSize: '16px',
    },
});

// Function to render all mermaid diagrams on the page
function renderMermaidDiagrams() {
    const diagrams = document.querySelectorAll('.mermaid-diagram');

    diagrams.forEach((element, index) => {
        const diagram = element.getAttribute('data-diagram') || element.textContent;
        if (!diagram) return;

        const id = `mermaid-${index}-${Math.random().toString(36).substr(2, 9)}`;

        mermaid.render(id, diagram).then(({ svg }) => {
            element.innerHTML = svg;
            element.classList.add('mermaid-rendered');
        }).catch((error) => {
            console.error('Failed to render mermaid diagram:', error);
            element.innerHTML = `<pre class="text-red-500">Error rendering diagram: ${error.message}</pre>`;
        });
    });
}

// Render on initial load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderMermaidDiagrams);
} else {
    renderMermaidDiagrams();
}

// Re-render on view transitions (for Astro)
document.addEventListener('astro:page-load', renderMermaidDiagrams);
