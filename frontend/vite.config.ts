import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  // Development server
  server: {
    port: 5173,
    strictPort: true,
    // Proxy API requests to the FastAPI backend during development.
    // The backend runs on localhost:8000 by default (python -m rasvcx).
    proxy: Object.fromEntries(
      // Every backend route prefix the UI calls (see src/api/client.ts).
      ['/health', '/ready', '/status', '/query', '/feedback', '/admin', '/eval', '/ingest'].map(
        (prefix) => [prefix, { target: 'http://localhost:8000', changeOrigin: true }],
      ),
    ),
  },

  // Production build output — served by FastAPI static files or CDN
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Chunk splitting: keep vendor (react) separate from app code
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },

  // Preview server (npm run preview) mirrors production
  preview: {
    port: 4173,
    strictPort: true,
  },
})