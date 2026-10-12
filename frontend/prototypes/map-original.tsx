import {useEffect, useMemo, useRef, useState} from 'react';
import * as maplibregl from 'maplibre-gl';
import mapWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';

// Resolve the worker as a Vite asset in both development and production.
maplibregl.setWorkerUrl(mapWorkerUrl);
import type {FeatureCollection, Point} from 'geojson';
import 'maplibre-gl/dist/maplibre-gl.css';

export type CrashRiskPoint = {
  latitude: number;
  longitude: number;
  riskProbability: number; // P(serious/fatal | recorded crash), 0–1
  crashCount?: number;
  roadway?: string;
  intersection?: string;
};
type Props = {points?: CrashRiskPoint[]; height?: number | string};

/** Design prototype. Pass model results from your Python backend, never HF tokens. */
export default function DeckGLMap({points = [], height = 600}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [layer, setLayer] = useState<'points' | 'heatmap'>('points');
  useEffect(() => {
    if (!container.current) return;
    let m: maplibregl.Map;
    try {
      m = new maplibregl.Map({container: container.current, center: [-84.388, 33.749], zoom: 11,
        style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'});
    } catch (e) {setError(e instanceof Error ? e.message : 'Map could not start'); return;}
    map.current = m;
    m.addControl(new maplibregl.NavigationControl(), 'top-right');
    const popup = new maplibregl.Popup({closeButton: true});
    m.on('error', () => setError('Some map resources could not load. Check your connection.'));
    m.on('load', () => {
      m.addSource('severity', {type: 'geojson', data: {type: 'FeatureCollection', features: []}});
      m.addLayer({id: 'severity-heatmap', type: 'heatmap', source: 'severity', layout: {visibility: 'none'},
        paint: {'heatmap-weight': ['get', 'probability'], 'heatmap-radius': 30, 'heatmap-opacity': .65}});
      m.addLayer({id: 'severity-points', type: 'circle', source: 'severity', paint: {
        'circle-radius': 7, 'circle-stroke-width': 1, 'circle-stroke-color': '#ffffff',
        'circle-color': ['interpolate', ['linear'], ['get', 'probability'], 0, '#48b7b0', .05, '#f2a541', .15, '#a82b45']}});
      m.on('click', 'severity-points', e => {
        const f = e.features?.[0];
        if (!f || f.geometry.type !== 'Point') return;
        const text = document.createElement('div');
        text.textContent = `${f.properties?.roadway || 'Selected location'} · Experimental conditional severity: ${(Number(f.properties?.probability) * 100).toFixed(2)}%`;
        popup.setLngLat(f.geometry.coordinates as [number, number]).setDOMContent(text).addTo(m);
      });
      m.on('mouseenter', 'severity-points', () => {m.getCanvas().style.cursor = 'pointer';});
      m.on('mouseleave', 'severity-points', () => {m.getCanvas().style.cursor = '';});
      setReady(true);
    });
    return () => {popup.remove(); m.remove(); map.current = null;};
  }, []);
  const valid = useMemo(() => points.filter(p => Number.isFinite(p.latitude) && Math.abs(p.latitude) <= 90 &&
    Number.isFinite(p.longitude) && Math.abs(p.longitude) <= 180 && Number.isFinite(p.riskProbability) &&
    p.riskProbability >= 0 && p.riskProbability <= 1), [points]);
  useEffect(() => {
    if (!ready || !map.current) return;
    const data: FeatureCollection<Point> = {type: 'FeatureCollection', features: valid.map(p => ({
      type: 'Feature', geometry: {type: 'Point', coordinates: [p.longitude, p.latitude]},
      properties: {probability: p.riskProbability, roadway: p.roadway ?? '', crashCount: p.crashCount ?? null}
    }))};
    (map.current.getSource('severity') as maplibregl.GeoJSONSource).setData(data);
  }, [valid, ready]);
  useEffect(() => {
    if (!ready || !map.current) return;
    map.current.setLayoutProperty('severity-points', 'visibility', layer === 'points' ? 'visible' : 'none');
    map.current.setLayoutProperty('severity-heatmap', 'visibility', layer === 'heatmap' ? 'visible' : 'none');
  }, [layer, ready]);
  return <section aria-label="Experimental severity map">
    <label>Display <select value={layer} onChange={e => setLayer(e.target.value as 'points' | 'heatmap')}>
      <option value="points">Probability points</option><option value="heatmap">Weighted density</option>
    </select></label>
    <p>Experimental severity given a crash. Use the same time scenario for location comparisons.</p>
    {layer === 'heatmap' && <p>Weighted density depends on the number and spacing of supplied points; it is not a probability surface.</p>}
    <div ref={container} style={{height, minHeight: 300, width: '100%'}} />
    {error && <p role="alert">{error}</p>}
    {!valid.length && <p role="status">No prediction points supplied. Pass backend results through the points prop.</p>}
    {valid.length < points.length && <p role="status">{points.length - valid.length} invalid points were omitted.</p>}
    <p>Point colors: teal 0%, amber 5%, burgundy 15% or higher. These are visual anchors, not validated risk categories.</p>
  </section>;
}
