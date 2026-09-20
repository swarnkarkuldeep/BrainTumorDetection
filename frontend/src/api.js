const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {}

export async function predict(file) {
  const formData = new FormData();
  formData.append("file", file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new ApiError(
      "Can't reach the scanning service. Check that the backend is running and try again."
    );
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : null;
    throw new ApiError(detail || "The scan couldn't be analyzed. Try a different file.");
  }

  return body;
}
