/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        base: '#0B0F14',
        panel: '#101620',
        'panel-raised': '#151C27',
        hairline: '#232B36',
        'hairline-soft': '#1A2129',
        ink: '#E7EBEF',
        'ink-dim': '#8B96A3',
        'ink-faint': '#4E5964',
        signal: '#3ED98B',
        'signal-dim': '#1F6B47',
        review: '#F2A93B',
        alert: '#EF5B54',
        wire: '#5B8DEF',
      },
      fontFamily: {
        display: ['Manrope', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      boxShadow: {
        none_: 'none',
      },
    },
  },
  plugins: [],
}
