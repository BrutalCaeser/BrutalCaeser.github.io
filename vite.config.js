import { defineConfig } from 'vite';

// User/organization GitHub Pages site (served at the domain root).
export default defineConfig({
  base: '/',
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0,
  },
});
