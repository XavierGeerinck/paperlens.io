import mermaid from 'mermaid';

/**
 * Diagrams follow the site palette: near-black ground, mint primary,
 * amber / azure / iris for the rest. See src/styles/global.css.
 */
const palette = {
    bg0: '#0d0f13',
    bg0h: '#080a0d',
    bg1: '#161a20',
    bg2: '#22272f',
    bg3: '#323944',
    fg: '#e6eaf0',
    fg2: '#98a2ae',
    green: '#35d492',
    blue: '#4c9ef5',
    purple: '#8b6cf2',
    amber: '#f5a623',
};

mermaid.initialize({
    startOnLoad: false,
    theme: 'base',
    fontFamily: "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
    themeVariables: {
        darkMode: true,
        fontSize: '14px',
        background: palette.bg0h,

        primaryColor: palette.bg1,
        primaryTextColor: palette.fg,
        primaryBorderColor: palette.green,
        secondaryColor: palette.bg1,
        secondaryTextColor: palette.fg,
        secondaryBorderColor: palette.blue,
        tertiaryColor: palette.bg0,
        tertiaryTextColor: palette.fg2,
        tertiaryBorderColor: palette.bg3,

        lineColor: palette.fg2,
        textColor: palette.fg,
        mainBkg: palette.bg1,
        nodeBorder: palette.bg3,
        nodeTextColor: palette.fg,
        clusterBkg: palette.bg0,
        clusterBorder: palette.bg2,
        titleColor: palette.fg,
        edgeLabelBackground: palette.bg0h,

        // sequence / state / gantt accents
        actorBkg: palette.bg1,
        actorBorder: palette.green,
        actorTextColor: palette.fg,
        signalColor: palette.fg2,
        signalTextColor: palette.fg2,
        labelBoxBkgColor: palette.bg1,
        labelBoxBorderColor: palette.bg3,
        labelTextColor: palette.fg,
        noteBkgColor: palette.bg1,
        noteBorderColor: palette.amber,
        noteTextColor: palette.fg,
        altBackground: palette.bg0,
        errorBkgColor: palette.bg1,
        errorTextColor: palette.fg,

        pie1: palette.green,
        pie2: palette.blue,
        pie3: palette.amber,
        pie4: palette.purple,
    },
});

// Function to render all mermaid diagrams on the page
function renderMermaidDiagrams() {
    const diagrams = document.querySelectorAll('.mermaid-diagram:not(.mermaid-rendered)');

    diagrams.forEach((element, index) => {
        const diagram = element.getAttribute('data-diagram') || element.textContent;
        if (!diagram) return;

        const id = `mermaid-${index}-${Math.random().toString(36).slice(2, 11)}`;

        mermaid
            .render(id, diagram)
            .then(({ svg }) => {
                element.innerHTML = svg;
                element.classList.add('mermaid-rendered');
            })
            .catch((error) => {
                console.error('Failed to render mermaid diagram:', error);
                element.innerHTML = `<p class="c-red">diagram error: ${error.message}</p>`;
                element.classList.add('mermaid-rendered');
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
