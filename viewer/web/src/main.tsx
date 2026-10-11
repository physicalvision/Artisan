import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";

import App from "@/App";
import "@/index.css";
import { viewerQueryClient } from "@/lib/query-client";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={viewerQueryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
