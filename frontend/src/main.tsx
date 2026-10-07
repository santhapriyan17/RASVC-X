/**
 * frontend/src/main.tsx
 *
 * Application entry point. Mounts the React tree into #root.
 *
 * Uses React 18 createRoot. StrictMode is enabled for development
 * warnings. No service workers, no analytics.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './styles.css'

const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error(
    'Root element #root not found in document. ' +
    'Ensure index.html contains <div id="root"></div>.'
  )
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
)
