import { lazy, Suspense, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import './App.css'

const OriginalMap = lazy(() => import('../prototypes/map-original'))
const ResearchMap = lazy(() => import('./map'))
const views = ['Original map', 'Research map'] as const

export default function App() {
  const [active, setActive] = useState(0)
  const tabs = useRef<(HTMLButtonElement | null)[]>([])

  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number
    if (event.key === 'ArrowRight') next = (index + 1) % views.length
    else if (event.key === 'ArrowLeft') next = (index + views.length - 1) % views.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = views.length - 1
    else return
    event.preventDefault()
    setActive(next)
    tabs.current[next]?.focus()
  }

  return <>
    <div className="map-view-switcher" role="tablist" aria-label="Map view">
      {views.map((label, index) => <button
        key={label}
        ref={node => { tabs.current[index] = node }}
        id={`map-tab-${index}`}
        type="button"
        role="tab"
        aria-selected={active === index}
        aria-controls={`map-panel-${index}`}
        tabIndex={active === index ? 0 : -1}
        onClick={() => setActive(index)}
        onKeyDown={event => navigate(event, index)}
      >{label}</button>)}
    </div>
    {views.map((label, index) => <div
      key={label}
      id={`map-panel-${index}`}
      role="tabpanel"
      aria-labelledby={`map-tab-${index}`}
      hidden={active !== index}
      tabIndex={0}
    >
      {active === index && <Suspense fallback={<p className="map-loading" role="status">Loading map…</p>}>
        {index === 0 ? <main className="original-map-view"><h1>Atlanta severity map</h1><OriginalMap height="65vh" /></main> : <ResearchMap />}
      </Suspense>}
    </div>)}
  </>
}
