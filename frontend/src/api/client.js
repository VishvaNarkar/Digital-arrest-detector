import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120_000, // 2 min — model inference can be slow
})

export async function analyzeText(text, threshold = 0.35) {
  const { data } = await api.post('/analyze/text', { text, threshold })
  return data
}

export async function analyzeAudio(file, lang = 'en-in', threshold = 0.35) {
  const form = new FormData()
  form.append('file', file)
  form.append('lang', lang)
  form.append('threshold', String(threshold))
  const { data } = await api.post('/analyze/audio', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function analyzeVideo(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/analyze/video', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}
