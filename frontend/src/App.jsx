import { useState, useRef, useCallback } from 'react'

export default function App() {
  const [state, setState] = useState('idle') // idle | processing | done | error
  const [originalPreview, setOriginalPreview] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [fileName, setFileName] = useState('')
  const [dragging, setDragging] = useState(false)
  const [queuePosition, setQueuePosition] = useState(null)
  const fileInputRef = useRef(null)

  const handleFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setErrorMsg('Please upload a valid image file.')
      setState('error')
      return
    }

    // Show original preview
    const preview = URL.createObjectURL(file)
    setOriginalPreview(preview)
    setFileName(file.name)
    setState('processing')
    setResultUrl(null)
    setErrorMsg('')

    try {
      // Check queue first
      try {
        const qRes = await fetch('/queue')
        const qData = await qRes.json()
        if (qData.pending_jobs > 0) {
          setQueuePosition(qData.pending_jobs + 1)
        }
      } catch { /* ignore */ }

      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/removebg', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || `Server error (${response.status})`)
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      setResultUrl(url)
      setQueuePosition(null)
      setState('done')
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong')
      setQueuePosition(null)
      setState('error')
    }
  }, [])

  const handleDownload = () => {
    if (!resultUrl) return
    const a = document.createElement('a')
    a.href = resultUrl
    const baseName = fileName.replace(/\.[^.]+$/, '')
    a.download = `${baseName}_no_bg.png`
    a.click()
  }

  const reset = () => {
    setState('idle')
    setOriginalPreview(null)
    setResultUrl(null)
    setErrorMsg('')
    setFileName('')
    setQueuePosition(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="#0a0a0a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          </div>
          <h1>BG Remover</h1>
        </div>
        <div className="status-badge">
          <span className="status-dot"></span>
          AI ready
        </div>
      </header>

      {/* Main */}
      <main className="main">
        <div className="title-section">
          <h2>Remove Backgrounds</h2>
          <p>Upload a photo and let AI remove the background in seconds.</p>
        </div>

        {/* ── Idle: Upload Zone ──────────────────────── */}
        {state === 'idle' && (
          <div
            className={`upload-zone ${dragging ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            id="upload-zone"
          >
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="upload-text">Drop an image here or click to browse</p>
            <p className="upload-hint">PNG, JPG, WEBP — max 10 MB</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="upload-input"
              onChange={(e) => handleFile(e.target.files?.[0])}
              id="file-input"
            />
          </div>
        )}

        {/* ── Processing ────────────────────────────── */}
        {state === 'processing' && (
          <div className="processing">
            <div className="spinner-container">
              <div className="spinner"></div>
            </div>
            <p className="processing-text">Removing background…</p>
            {queuePosition && (
              <span className="queue-info">Queue position: #{queuePosition}</span>
            )}
          </div>
        )}

        {/* ── Error ─────────────────────────────────── */}
        {state === 'error' && (
          <>
            <div className="error">
              <span className="error-text">{errorMsg}</span>
              <button className="error-dismiss" onClick={reset}>×</button>
            </div>
          </>
        )}

        {/* ── Result ────────────────────────────────── */}
        {state === 'done' && resultUrl && (
          <div className="result">
            <div className="result-images">
              {/* Original */}
              <div className="image-card">
                <div className="image-card-header">
                  <span className="image-card-dot dot-original"></span>
                  <span className="image-card-label">Original</span>
                </div>
                <div className="image-wrapper">
                  <img src={originalPreview} alt="Original" />
                </div>
              </div>
              {/* Result */}
              <div className="image-card">
                <div className="image-card-header">
                  <span className="image-card-dot dot-result"></span>
                  <span className="image-card-label">Background Removed</span>
                </div>
                <div className="image-wrapper checkerboard">
                  <img src={resultUrl} alt="Background removed" />
                </div>
              </div>
            </div>

            <div className="actions">
              <button className="btn btn-primary" onClick={handleDownload} id="download-btn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download PNG
              </button>
              <button className="btn btn-secondary" onClick={reset} id="new-upload-btn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="1 4 1 10 7 10" />
                  <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                </svg>
                New Image
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by DeepLabV3+ · PyTorch</p>
      </footer>
    </div>
  )
}
