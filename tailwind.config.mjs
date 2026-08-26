import typography from '@tailwindcss/typography';

/**
 * PaperLens — "TUI" design system.
 *
 * One typeface (Geist Mono), one primary (mint), four semantic hues.
 * The neutral scales (slate/zinc/gray/neutral/stone) are all remapped onto the
 * cool near-black ramp so existing simulation components inherit the new ground
 * without being rewritten. See docs/mockups/brand.html for the roles.
 */

const mono = [
  'Geist Mono',
  'IBM Plex Mono',
  'JetBrains Mono',
  'ui-monospace',
  'SFMono-Regular',
  'Menlo',
  'monospace',
];

// ground → ink, dark first: 950/900 surfaces, 800/700 borders, 500/400 muted, 300/100 bright
const neutral = {
  50: '#f4f6f9',
  100: '#e6eaf0', // --fg
  200: '#d5dbe3',
  300: '#c6cdd6', // --fg1
  400: '#98a2ae', // --fg2
  500: '#6b7583', // --fg4
  600: '#4b5563', // --gray
  700: '#323944', // --bg3
  800: '#22272f', // --bg2
  900: '#161a20', // --bg1
  950: '#0d0f13', // --bg0
};

const mint = {
  50: '#e8fbf3',
  100: '#c6f5e2',
  200: '#9deed0',
  300: '#66e3b5',
  400: '#35d492', // primary
  500: '#1fbc7d',
  600: '#159a66',
  700: '#12784f',
  800: '#0e5539',
  900: '#0b3a28',
  950: '#06231a',
};

const amber = {
  50: '#fef6e7',
  100: '#fce7bd',
  200: '#fad693',
  300: '#f8c05a',
  400: '#f5a623',
  500: '#dc8c0c',
  600: '#b5710a',
  700: '#8c5708',
  800: '#633e06',
  900: '#432a05',
  950: '#291a03',
};

const azure = {
  50: '#eaf3fe',
  100: '#cbe2fc',
  200: '#a8cff9',
  300: '#7bb7f7',
  400: '#4c9ef5',
  500: '#2b86e8',
  600: '#1b6ac2',
  700: '#15529a',
  800: '#103c70',
  900: '#0c2a4e',
  950: '#081a30',
};

const aqua = {
  50: '#e9f7ff',
  100: '#c9ecff',
  200: '#a3dfff',
  300: '#7dd4ff',
  400: '#5cc8ff',
  500: '#33b0f0',
  600: '#1e8dc7',
  700: '#176d9b',
  800: '#114e6f',
  900: '#0c364d',
  950: '#07212f',
};

const iris = {
  50: '#f1edfe',
  100: '#ded4fc',
  200: '#c7b6f9',
  300: '#a992f6',
  400: '#8b6cf2',
  500: '#7452e6',
  600: '#5d3ec6',
  700: '#48309b',
  800: '#342270',
  900: '#24184e',
  950: '#160f30',
};

const magenta = {
  50: '#fdeefe',
  100: '#f9d1fb',
  200: '#f2b0f6',
  300: '#e694f2',
  400: '#d977f0',
  500: '#c257db',
  600: '#a03fb6',
  700: '#7c318d',
  800: '#592365',
  900: '#3d1846',
  950: '#250e2b',
};

const danger = {
  50: '#fdecec',
  100: '#fbcfd1',
  200: '#f8adb1',
  300: '#f78289',
  400: '#f5555d',
  500: '#e13540',
  600: '#bb242f',
  700: '#911c25',
  800: '#67151b',
  900: '#470f13',
  950: '#2b090b',
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        // one family, everywhere
        sans: mono,
        mono,
        heading: mono,
        sketch: mono,
      },
      colors: {
        // semantic names (preferred for new code)
        bg0: '#0d0f13',
        bg0h: '#080a0d',
        bg1: '#161a20',
        bg2: '#22272f',
        bg3: '#323944',
        ink: '#e6eaf0',
        ink1: '#c6cdd6',
        ink2: '#98a2ae',
        ink4: '#6b7583',
        mute: '#4b5563',
        onhue: '#0b0d11',

        // hues, by role
        mint,
        amber,
        azure,
        aqua,
        iris,
        magenta,
        danger,

        // legacy families, remapped so existing components stay on-brand
        slate: neutral,
        zinc: neutral,
        gray: neutral,
        neutral,
        stone: neutral,
        emerald: mint,
        green: mint,
        teal: mint,
        lime: mint,
        yellow: amber,
        orange: amber,
        blue: azure,
        sky: azure,
        cyan: aqua,
        indigo: iris,
        violet: iris,
        purple: iris,
        fuchsia: magenta,
        pink: magenta,
        rose: danger,
        red: danger,
      },
      borderRadius: {
        DEFAULT: '4px',
        sm: '3px',
        md: '4px',
        lg: '6px',
        xl: '6px',
        '2xl': '8px',
      },
      gridTemplateColumns: {
        16: 'repeat(16, minmax(0, 1fr))',
        32: 'repeat(32, minmax(0, 1fr))',
      },
    },
  },
  plugins: [typography],
};
