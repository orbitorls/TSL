import type { Artifact, InferenceSettings, Track, TrackKey } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || res.statusText);
  }
  return res.json() as Promise<T>;
}

export function getApiBase(): string {
  return API_BASE;
}

export function getWsUrl(sessionId: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws/session/${sessionId}`;
}

export async function fetchTracks(): Promise<Track[]> {
  return api<Track[]>("/api/tracks");
}

export async function fetchArtifacts(track: TrackKey): Promise<Artifact[]> {
  return api<Artifact[]>(`/api/artifacts?track=${track}`);
}

export async function createSession(track: TrackKey): Promise<{ session_id: string; track: string }> {
  return api("/api/session", {
    method: "POST",
    body: JSON.stringify({ track }),
  });
}

export async function loadModel(
  sessionId: string,
  artifactName: string
): Promise<{ ok: boolean; backend: string; num_classes: number; load_time_ms: number }> {
  return api(`/api/session/${sessionId}/load`, {
    method: "POST",
    body: JSON.stringify({ artifact_name: artifactName }),
  });
}

export async function patchSettings(sessionId: string, settings: Partial<InferenceSettings>): Promise<void> {
  await api(`/api/session/${sessionId}/settings`, {
    method: "PATCH",
    body: JSON.stringify(settings),
  });
}

export async function transcriptAction(
  sessionId: string,
  action: "space" | "backspace" | "clear"
): Promise<{ text: string }> {
  return api(`/api/session/${sessionId}/transcript`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}
