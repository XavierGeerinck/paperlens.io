import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        heading: ['Space Grotesk', 'sans-serif'],
        sketch: ['Caveat', 'cursive'],
      },
      gridTemplateColumns: {
        '16': 'repeat(16, minmax(0, 1fr))',
        '32': 'repeat(32, minmax(0, 1fr))',
      },
    },
  },
  plugins: [typography],
};
