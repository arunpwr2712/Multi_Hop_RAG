import { ChatResponse, GraphResponse, ReadinessResponse } from "@/types/rag";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function readApiError(response: Response, fallbackMessage: string): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: string };
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail;
    }
  } catch {
    // ignore JSON parse failures and use text fallback
  }

  const text = await response.text();
  return text || fallbackMessage;
}

export async function checkReadiness(): Promise<ReadinessResponse> {
  const response = await fetch(`${API_BASE_URL}/api/readiness`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await readApiError(response, "Failed to check server readiness"));
  }

  return (await response.json()) as ReadinessResponse;
}

export async function sendChat(query: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(await readApiError(response, "Failed to fetch chat response"));
  }

  return (await response.json()) as ChatResponse;
}

export async function fetchGraph(): Promise<GraphResponse> {
  const response = await fetch(`${API_BASE_URL}/api/graph`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await readApiError(response, "Failed to fetch graph"));
  }

  return (await response.json()) as GraphResponse;
}

export async function resetSession(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/reset`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await readApiError(response, "Failed to reset session"));
  }
}
