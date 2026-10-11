import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const apiHost = process.env.ARTISAN_VIEWER_API_HOST ?? "127.0.0.1";
const apiPort = process.env.ARTISAN_VIEWER_API_PORT ?? "8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three"],
          ui: ["lucide-react", "class-variance-authority", "clsx", "tailwind-merge"],
        },
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": `http://${apiHost}:${apiPort}`,
      "/health": `http://${apiHost}:${apiPort}`,
    },
  },
});
