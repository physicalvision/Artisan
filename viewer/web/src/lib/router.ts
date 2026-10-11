export type AppRoute = { page: "viewer" } | { page: "chat" };

export function parseRoute(): AppRoute {
  return { page: window.location.pathname.startsWith("/chat") ? "chat" : "viewer" };
}

export function navigateTo(route: AppRoute): void {
  const path = route.page === "chat" ? "/chat" : "/viewer";
  if (window.location.pathname === path) return;
  window.history.pushState(null, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}
