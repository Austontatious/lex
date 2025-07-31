import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dynamically pick backend port from env or fallback to 8000
const backendPort = process.env.REACT_APP_BACKEND_PORT || '8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: parseInt(process.env.PORT || '3000'),
    proxy: {
      '/process': `http://localhost:${backendPort}`,
      '/models': `http://localhost:${backendPort}`,
      '/ws': {
        target: `ws://localhost:${backendPort}`,
        ws: true
      }
    }
  }
})

