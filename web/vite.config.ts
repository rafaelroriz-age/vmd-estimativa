import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  base: './',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg', 'data/*.json', 'data/*.geojson'],
      manifest: {
        name: 'VMD por SRE — como estimamos',
        short_name: 'VMD SRE',
        description: 'Storytelling dos modelos GLMM-EB e GLMM-CAR e mapa interativo das SREs de Goiás.',
        lang: 'pt-BR', theme_color: '#1f6fd1', background_color: '#fcfcfb', display: 'standalone',
        icons: [
          { src: 'icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' },
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,json,geojson,woff2}'],
        maximumFileSizeToCacheInBytes: 8 * 1024 * 1024,
        runtimeCaching: [{
          urlPattern: /^https:\/\/[a-d]\.basemaps\.cartocdn\.com\/.*/,
          handler: 'StaleWhileRevalidate',
          options: { cacheName: 'basemap', expiration: { maxEntries: 400, maxAgeSeconds: 60 * 60 * 24 * 30 } },
        }],
      },
    }),
  ],
  build: { chunkSizeWarningLimit: 1200 },
})
