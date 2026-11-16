import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the UI runs on Vite's server and forwards API calls to the backend.
const apiTarget = process.env.VITE_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": apiTarget },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
