import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import YouTubePredictorDashboard from './youtube_predictor.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <YouTubePredictorDashboard />
  </StrictMode>,
)
