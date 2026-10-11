import type { RecordSummary } from "@/lib/types";

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

export function fetchRecords(query: string, offset: number, signal?: AbortSignal): Promise<RecordSummary[]> {
  return getJson(`/api/records?q=${encodeURIComponent(query)}&offset=${offset}&limit=100`, signal);
}

export function fetchRecordSummary(recordId: string, signal?: AbortSignal): Promise<RecordSummary> {
  return getJson(`/api/records/${encodeURIComponent(recordId)}/summary`, signal);
}

export async function fetchRecordText(recordId: string, name: string, signal?: AbortSignal): Promise<string> {
  const result = await getJson<{content: string}>(`/api/records/${encodeURIComponent(recordId)}/text/${encodeURIComponent(name)}?full=true`, signal);
  return result.content;
}
