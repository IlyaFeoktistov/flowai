import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Бэкенд — src/main.py (uvicorn main:app, дефолтный порт 8000, см.
// docs/web-ui.md) — проксируем /api и /ws-хендшейк на него, чтобы фронт
// стучался по относительным путям и не думал про CORS/порт.
const BACKEND = 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    // Держать в синхроне с tsconfig.app.json's paths — TS резолвит типы,
    // Vite резолвит реальные модули, оба должны понимать один и тот же @.
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true, ws: true },
    },
  },
})
