import { useState } from 'react'
import data from './research-data.json'

type Counts = { crashes: number; severe: number }
type Cell = { id: string; x: number; y: number; latitude: number; longitude: number; years: Record<string, Counts> }
const cells = data.cells as Cell[]
const percent = (n: number) => `${(100 * n).toFixed(2)}%`
function counts(c: Cell, year: string): Counts {
  return Object.entries(c.years).filter(([y]) => year === 'all' || y === year).reduce((a, [, b]) => ({ crashes: a.crashes + b.crashes, severe: a.severe + b.severe }), { crashes: 0, severe: 0 })
}
function interval(n: number, k: number) {
  const z = 1.96, rate = k / n, den = 1 + z * z / n
  const center = (rate + z * z / (2 * n)) / den
  const half = z * Math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / den
  return `${percent(Math.max(0, center - half))} – ${percent(Math.min(1, center + half))}`
}
const support = (c: Counts) => c.crashes >= 100 && c.severe >= 10
export default function ResearchMap() {
  const [year, setYear] = useState('all')
  const [layer, setLayer] = useState('support')
  const [selected, setSelected] = useState('741_3735')
  const [zoom, setZoom] = useState(1)
  const visible = cells.map(c => ({ ...c, ...counts(c, year) })).filter(c => c.crashes > 0)
  const active = visible.find(c => c.id === selected)
  const totals = visible.reduce((a, c) => ({ crashes: a.crashes + c.crashes, severe: a.severe + c.severe }), { crashes: 0, severe: 0 })
  const enough = visible.filter(support).length
  const width = data.maxX - data.minX, height = data.maxY - data.minY
  const sx = (x: number) => (x - data.minX) * 20 + 12
  const sy = (y: number) => (data.maxY - y) * 20 + 12
  const color = (c: Counts) => {
    if (!support(c)) return '#aebbc9'
    if (layer === 'support') return '#147b85'
    const rate = c.severe / c.crashes
    return rate < .01 ? '#b9e4e4' : rate < .02 ? '#62b8c2' : rate < .03 ? '#258b9d' : rate < .05 ? '#175773' : '#102f50'
  }
  return <main>
    <header><div><p className="eyebrow">ATLANTA · COLLISION RESEARCH</p><h1>Where the evidence is.</h1><p>Explore historical serious and fatal outcomes, one square kilometer at a time.</p></div><span className="badge">Research use · 2021–2024</span></header>
    <section className="stats" aria-label="Selected period summary"><div><span>Recorded crashes</span><strong>{totals.crashes.toLocaleString()}</strong></div><div><span>Serious / fatal crashes</span><strong>{totals.severe.toLocaleString()}</strong></div><div><span>Observed severe share</span><strong>{percent(totals.severe / totals.crashes)}</strong></div><div><span>Cells with more evidence</span><strong>{enough} <small>/ {visible.length}</small></strong></div></section>
    <section className="workspace">
      <div className="map-panel">
        <div className="toolbar"><label>Period<select value={year} onChange={e => setYear(e.target.value)}><option value="all">2021–2024 combined</option>{data.years.map(y => <option key={y}>{y}</option>)}</select></label><label>Color by<select value={layer} onChange={e => setLayer(e.target.value)}><option value="support">Historical evidence</option><option value="rate">Observed severe share</option></select></label><div className="zoom"><button aria-label="Zoom out" disabled={zoom === 1} onClick={() => setZoom(z => Math.max(1, z - .5))}>−</button><button aria-label="Zoom in" disabled={zoom === 3} onClick={() => setZoom(z => Math.min(3, z + .5))}>+</button></div></div>
        <div className="map-scroll"><svg aria-label="Atlanta historical evidence grid. Use the area selector to access each cell by keyboard." role="img" viewBox={`0 0 ${width * 20 + 24} ${height * 20 + 24}`} style={{ width: `${zoom * 100}%`, minWidth: '360px' }}>
          <rect width="100%" height="100%" fill="#edf2f5" />
          {data.roads.map((path, i) => <polyline key={i} points={path.map(([x,y]) => `${sx(x)},${sy(y)}`).join(' ')} fill="none" stroke="#c4cfd8" strokeWidth="1.3" />)}
          {visible.map(c => <rect key={c.id} x={sx(c.x)} y={sy(c.y + 1)} width="19" height="19" rx="1.5" fill={color(c)} fillOpacity=".86" stroke={c.id === selected ? '#ed7c35' : '#ffffff'} strokeWidth={c.id === selected ? 2.6 : .5} onClick={() => setSelected(c.id)} style={{ cursor: 'pointer' }}><title>{c.id}: {c.crashes} crashes, {c.severe} serious/fatal. {support(c) ? 'More historical evidence' : 'Limited evidence'}</title></rect>)}
          <text x="16" y="30" fontSize="14" fill="#173550">N ↑</text>
          <path d={`M 18 ${height * 20 + 10} h 40`} stroke="#173550" strokeWidth="2" /><text x="63" y={height * 20 + 14} fontSize="12" fill="#173550">2 km</text>
        </svg></div>
        <div className="legend"><span><i style={{ background: '#aebbc9' }} />Limited evidence · share color withheld</span>{layer === 'support' ? <span><i style={{ background: '#147b85' }} />More historical evidence</span> : <span className="ramp">Severe share: &lt;1% <b /> ≥5%</span>}</div>
        <p className="map-note">Blank areas have no records in this export. They are not evidence of safety. Grid boundaries do not follow roads or neighborhoods.</p>
      </div>
      <aside><label className="area-label">Inspect an area<select value={active?.id ?? ''} onChange={e => setSelected(e.target.value)}><option value="">Select a grid cell</option>{visible.map(c => <option key={c.id} value={c.id}>{c.id} · {c.latitude.toFixed(3)}, {c.longitude.toFixed(3)}</option>)}</select></label>
        {active ? <div aria-live="polite"><p className="eyebrow">1 KM² GRID CELL</p><h2>{active.latitude.toFixed(3)}° N<br />{Math.abs(active.longitude).toFixed(3)}° W</h2><span className={`status ${support(active) ? 'supported' : ''}`}>{support(active) ? 'More historical evidence' : 'Limited evidence'}</span><dl><div><dt>Recorded crashes</dt><dd>{active.crashes.toLocaleString()}</dd></div><div><dt>Serious / fatal</dt><dd>{active.severe}</dd></div><div><dt>Observed severe share</dt><dd>{percent(active.severe / active.crashes)}</dd></div><div><dt>Approx. 95% interval</dt><dd>{interval(active.crashes, active.severe)}</dd></div></dl><p className="detail-note">{support(active) ? 'This area meets the historical-count screen. That does not establish reliable model predictions.' : 'Too few recorded crashes or severe outcomes meet the historical-count screen. No risk score is assigned.'}</p><p className="detail-note">Coordinates mark the cell center, not a crash location. The interval describes an observed proportion, not a model prediction.</p></div> : <div className="empty"><h2>Select a recorded area</h2><p>Choose a colored cell or use the area selector. A cell selected in another period may have no records here.</p></div>}
        <div className="model-note"><strong>Model scores are not displayed</strong><p>The current model has not established dependable location-level probabilities.</p></div>
      </aside>
    </section>
    <section className="method"><h2>Read this as severity given a recorded crash.</h2><p>The percentage is the share of recorded crashes resulting in serious or fatal injury. It is not the chance of a crash occurring at a location. Counts also reflect traffic volume and reporting.</p><details><summary>Evidence thresholds, uncertainty and sources</summary><p>“More historical evidence” requires at least 100 crashes and 10 serious/fatal outcomes in the selected period. This is a descriptive screen, not a validated reliability threshold. All other cells remain gray; zero severe outcomes does not mean zero risk.</p><p>Intervals use the Wilson binomial method. Spatial and temporal dependence can make uncertainty larger than shown. Selected-period counts are historical summaries, not training or forecast validation.</p><p>Crash source: the existing Atlanta development dataset, 2021–2024. The 2025 holdout and Kaggle are excluded. Reference linework: <a href="https://gis.atlantaga.gov/dpcd/rest/services/OpenDataService1/FeatureServer/33" target="_blank" rel="noreferrer">City of Atlanta StreetCenterline</a>; current reference geometry does not establish historical road attributes. Grid projection: UTM zone 16N.</p></details></section>
  </main>
}
