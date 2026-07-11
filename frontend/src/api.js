// Use relative URL so Vite dev server proxies /api to the backend (no CORS issues)
const API_BASE = '';

export async function getStockData(ticker, period = '2y') {
  const res = await fetch(
    `${API_BASE}/api/stock/${encodeURIComponent(ticker)}?period=${period}`
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to fetch stock data');
  }
  return res.json();
}

export async function getPrediction(ticker, model, steps = 30) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker, model, steps }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.ok;
}
