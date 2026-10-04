import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  build: {
    rollupOptions: {
      output: {
        // Split large third-party libraries into separate chunks so the main
        // application bundle stays small and vendor code can be cached across
        // deploys. This only changes how the build is grouped on disk — the
        // runtime behaviour of the application is unchanged.
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'charts-vendor': ['recharts'],
        },
      },
    },
  },
})
