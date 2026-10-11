import { lazy, Suspense, type JSX } from "react";
import { useRoute } from "@/lib/useRoute";
import { TooltipProvider } from "@/components/ui/tooltip";

const ViewerShell = lazy(() => import("@/ViewerShell"));
const ChatShell = lazy(() => import("@/ChatShell"));

export default function App(): JSX.Element {
  const route = useRoute();
  return (
    <TooltipProvider>
      <div className="flex h-screen flex-col">
        <Suspense fallback={<p className="p-4 text-sm">Loading…</p>}>
          {route.page === "chat" ? <ChatShell /> : <ViewerShell />}
        </Suspense>
      </div>
    </TooltipProvider>
  );
}
