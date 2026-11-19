export const isDark = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches
// Paleta validada com validate_palette.js (CVD ΔE ≥ 22, contraste ≥ 3:1) nos dois temas.
export const pal = (dark: boolean) => dark
  ? { count: '#4a92ea', est: '#e8574d', none: '#6b6b68', up: '#e8a34d', down: '#3fb3a8', surface: '#1a1a19', ink: '#ecebe8' }
  : { count: '#1f6fd1', est: '#d23a31', none: '#a9a8a3', up: '#c2570f', down: '#0f8a8a', surface: '#fcfcfb', ink: '#1c1c1a' }
