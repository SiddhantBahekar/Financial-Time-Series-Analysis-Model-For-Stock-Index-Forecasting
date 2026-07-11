import { useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { getStockData, getPrediction, healthCheck } from './api';
import './App.css';

const SUGGESTED_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NIFTY 50'];

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [model, setModel] = useState('auto');
  const [steps, setSteps] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [backendOk, setBackendOk] = useState(null);

  const checkBackend = useCallback(async () => {
    try {
      const ok = await healthCheck();
      setBackendOk(ok);
      return ok;
    } catch {
      setBackendOk(false);
      return false;
    }
  }, []);

  const loadHistory = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const data = await getStockData(ticker);
      const points = data.dates.map((d, i) => ({
        date: d,
        historical: data.values[i],
        forecast: null,
      }));
      setChartData({ points, ticker: data.ticker, model: null, stats: null });
    } catch (e) {
      setError(e.message);
      setChartData(null);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  const runPrediction = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const ok = await checkBackend();
      if (!ok) {
        setError('Backend not running. Start it with: cd backend && uvicorn main:app --reload');
        setLoading(false);
        return;
      }
      const result = await getPrediction(ticker, model, steps);
      const historyMap = new Map(
        result.historical_dates.map((d, i) => [d, { date: d, historical: result.historical_values[i], forecast: null }])
      );
      result.forecast_dates.forEach((d, i) => {
        if (!historyMap.has(d)) historyMap.set(d, { date: d, historical: null, forecast: result.forecast_values[i] });
        else historyMap.get(d).forecast = result.forecast_values[i];
      });
      const points = [...historyMap.entries()]
        .sort((a, b) => a[0].localeCompare(b[0]))
        .map(([, v]) => v);
      setChartData({ points, ticker: result.ticker, model: result.model, stats: result.stats || null });
    } catch (e) {
      setError(e.message);
      setChartData(null);
    } finally {
      setLoading(false);
    }
  }, [ticker, model, steps, checkBackend]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-5 flex flex-wrap items-center justify-between gap-4">
          <h1 className="text-xl font-semibold tracking-tight text-accent-emerald">
            Stock Price Prediction
          </h1>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className={`inline-block w-2 h-2 rounded-full ${backendOk === true ? 'bg-emerald-500' : backendOk === false ? 'bg-red-500' : 'bg-slate-500'}`} />
            {backendOk === true ? 'API connected' : backendOk === false ? 'API offline' : 'Check API'}
            <button
              type="button"
              onClick={checkBackend}
              className="text-cyan-400 hover:text-cyan-300 ml-1"
            >
              Refresh
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <section className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 mb-6">
          <h2 className="text-lg font-medium text-slate-200 mb-4">Inputs</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Ticker</label>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g. AAPL"
                className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-accent-emerald focus:border-transparent"
              />
              <div className="flex flex-wrap gap-1 mt-1">
                {SUGGESTED_TICKERS.map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => {
                      if(t === 'NIFTY 50'){
                      setTicker('^NSEI');
                      } else {
                        setTicker(t);
                      }
                    } 
                  }
                    className="text-xs px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-slate-100 focus:ring-2 focus:ring-accent-emerald focus:border-transparent"
              >
                <option value="auto">Auto (best)</option>
                <option value="gbrt">GBRT (fast + accurate)</option>
                <option value="arima">ARIMA</option>
                <option value="lstm">LSTM</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Forecast days</label>
              <input
                type="number"
                min={5}
                max={90}
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value) || 30)}
                className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-slate-100 focus:ring-2 focus:ring-accent-emerald focus:border-transparent"
              />
            </div>
            <div className="flex items-end gap-2">
              <button
                type="button"
                onClick={loadHistory}
                disabled={loading}
                className="flex-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 px-4 py-2 text-sm font-medium disabled:opacity-50"
              >
                History
              </button>
              <button
                type="button"
                onClick={runPrediction}
                disabled={loading}
                className="flex-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
              >
                {loading ? '…' : 'Predict'}
              </button>
            </div>
          </div>
          {error && (
            <p className="mt-3 text-sm text-red-400 bg-red-900/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}
        </section>

        {chartData && (
          <section className="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
            <h2 className="text-lg font-medium text-slate-200 mb-2">
              {chartData.ticker}
              {chartData.model && (
                <span className="text-slate-400 font-normal ml-2">
                  ({chartData.model.toUpperCase()} forecast)
                </span>
              )}
            </h2>

            {chartData.stats && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-5">
                <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-4">
                  <div className="text-sm text-slate-400 mb-2">Market stats (history)</div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    <div className="text-slate-400">Last close</div>
                    <div className="text-slate-100 text-right">{chartData.stats.series?.last_close?.toFixed?.(2) ?? '—'}</div>
                    <div className="text-slate-400">Change</div>
                    <div className="text-slate-100 text-right">{chartData.stats.series?.change_pct != null ? `${chartData.stats.series.change_pct.toFixed(2)}%` : '—'}</div>
                    <div className="text-slate-400">Daily vol</div>
                    <div className="text-slate-100 text-right">{chartData.stats.series?.daily_volatility_pct != null ? `${chartData.stats.series.daily_volatility_pct.toFixed(2)}%` : '—'}</div>
                    <div className="text-slate-400">Max drawdown</div>
                    <div className="text-slate-100 text-right">{chartData.stats.series?.max_drawdown_pct != null ? `${chartData.stats.series.max_drawdown_pct.toFixed(2)}%` : '—'}</div>
                    <div className="text-slate-400">52w high</div>
                    <div className="text-slate-100 text-right">{chartData.stats.series?.high_52w?.toFixed?.(2) ?? '—'}</div>
                    <div className="text-slate-400">52w low</div>
                    <div className="text-slate-100 text-right">{chartData.stats.series?.low_52w?.toFixed?.(2) ?? '—'}</div>
                  </div>
                </div>

                <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-4">
                  <div className="text-sm text-slate-400 mb-2">Backtest (holdout)</div>
                  {chartData.stats.backtest ? (
                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                      <div className="text-slate-400">RMSE</div>
                      <div className="text-slate-100 text-right">{chartData.stats.backtest.rmse?.toFixed?.(3) ?? '—'}</div>
                      <div className="text-slate-400">MAE</div>
                      <div className="text-slate-100 text-right">{chartData.stats.backtest.mae?.toFixed?.(3) ?? '—'}</div>
                      <div className="text-slate-400">MAPE</div>
                      <div className="text-slate-100 text-right">{chartData.stats.backtest.mape_pct != null ? `${chartData.stats.backtest.mape_pct.toFixed(2)}%` : '—'}</div>
                      <div className="text-slate-400">Directional acc</div>
                      <div className="text-slate-100 text-right">
                        {chartData.stats.backtest.directional_accuracy_pct != null ? `${chartData.stats.backtest.directional_accuracy_pct.toFixed(1)}%` : '—'}
                      </div>
                      <div className="text-slate-400">R²</div>
                      <div className="text-slate-100 text-right">{chartData.stats.backtest.r2 != null ? chartData.stats.backtest.r2.toFixed(3) : '—'}</div>
                      <div className="text-slate-400">Window</div>
                      <div className="text-slate-100 text-right">
                        {chartData.stats.backtest.test_window?.start && chartData.stats.backtest.test_window?.end
                          ? `${chartData.stats.backtest.test_window.start} → ${chartData.stats.backtest.test_window.end}`
                          : '—'}
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-500">Backtest unavailable for this model/ticker.</div>
                  )}
                </div>

                <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-4">
                  <div className="text-sm text-slate-400 mb-2">Forecast summary</div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    <div className="text-slate-400">Mean</div>
                    <div className="text-slate-100 text-right">{chartData.stats.forecast?.forecast_mean?.toFixed?.(2) ?? '—'}</div>
                    <div className="text-slate-400">Min</div>
                    <div className="text-slate-100 text-right">{chartData.stats.forecast?.forecast_min?.toFixed?.(2) ?? '—'}</div>
                    <div className="text-slate-400">Max</div>
                    <div className="text-slate-100 text-right">{chartData.stats.forecast?.forecast_max?.toFixed?.(2) ?? '—'}</div>
                    <div className="text-slate-400">Std</div>
                    <div className="text-slate-100 text-right">{chartData.stats.forecast?.forecast_std?.toFixed?.(2) ?? '—'}</div>
                    <div className="text-slate-400">Residual σ</div>
                    <div className="text-slate-100 text-right">{chartData.stats.forecast?.residual_std?.toFixed?.(2) ?? '—'}</div>
                  </div>
                  <div className="text-xs text-slate-500 mt-2">
                    Backtest/intervals are approximate and do not guarantee future accuracy.
                  </div>
                </div>
              </div>
            )}

            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData.points}
                  margin={{ top: 10, right: 20, left: 10, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="date"
                    stroke="#94a3b8"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v) => v.slice(0, 7)}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0))}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#94a3b8' }}
                    formatter={(value, name) => [value != null ? value.toFixed(2) : '—', name]}
                    labelFormatter={(label) => label}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="historical"
                    name="Historical"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}
      </main>

      {/* <footer className="max-w-6xl mx-auto px-4 py-6 text-center text-sm text-slate-500 border-t border-slate-800 mt-8">
        Stock data from Yahoo Finance. Predictions use ARIMA and LSTM models — for education only, not financial advice.
      </footer> */}
    </div>
  );
}

export default App;
