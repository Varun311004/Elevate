// Configuration module
import './runtime-config.js';

function getDefaultApiBase() {
  if (typeof window === "undefined") {
    return "http://127.0.0.1:5000/api";
  }

  const { hostname, port, protocol } = window.location;

  // Local development frontend (python http.server, Vite, etc.)
  if (
    hostname === "127.0.0.1" ||
    hostname === "localhost"
  ) {
    return "http://127.0.0.1:5000/api";
  }

  // Production
  return `${protocol}//${window.location.host}/api`;
}

const SAME_ORIGIN_API = getDefaultApiBase();

function normalizeApiBase(url) {
  return String(url || '').trim().replace(/\/+$/, '');
}

function resolveRuntimeApiBase() {
  if (typeof window === 'undefined') {
    return '';
  }

  try {
    const search = new URLSearchParams(window.location.search || '');
    const queryValue = search.get('api_base') || search.get('apiBase');
    if (queryValue) {
      const normalized = normalizeApiBase(queryValue);
      if (normalized) {
        localStorage.setItem('ELEVATE_API_BASE_URL', normalized);
      }
    }

    const stored = localStorage.getItem('ELEVATE_API_BASE_URL');
    return normalizeApiBase(stored);
  } catch (_) {
    return '';
  }
}

const RUNTIME_API_BASE = resolveRuntimeApiBase();
const GLOBAL_API_BASE = normalizeApiBase(
  (typeof window !== 'undefined' && (
    window.__ELEVATE_API_BASE_URL__
    || window.ELEVATE_RUNTIME_CONFIG?.API_BASE_URL
  )) || ''
);

export const config = {
  // API Configuration (will be used when backend is ready)
  API_BASE_URL:
    RUNTIME_API_BASE
    || GLOBAL_API_BASE
    || import.meta?.env?.VITE_API_BASE_URL
    || SAME_ORIGIN_API,
  API_REQUEST_TIMEOUT_MS: 30000,
  API_TEST_CREATE_TIMEOUT_MS: 180000,
  
  // Camera and Emotion Detection
  EMOTION_DETECTION_INTERVAL: 500, // ms
  BREAK_TIME: 120, // seconds
  MODEL_LOAD_RETRY_MAX: 3,
  EMOTION_ALLOW_SIMULATION_FALLBACK: false,
  EMOTION_TFJS_MODEL_URL: '/js/emotion_tfjs/model.json',
  EMOTION_TFJS_EMOTION_HEAD_URL: '',
  EMOTION_TFJS_ENGAGEMENT_HEAD_URL: '',
  EMOTION_PREFER_SERVER_INFERENCE: false,
  
  // Model URLs (face-api.js CDN)
  MODEL_URLS: [
    'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/',
    'https://justadudewhohacks.github.io/face-api.js/models/',
    '/models/'
  ],
  
  // Question timing
  DEFAULT_QUESTION_TIME: 30, // seconds
  
  // Emotion thresholds
  EMOTION_CONFIDENCE_THRESHOLD: 0.5,
  FRUSTRATION_THRESHOLD: 3, // consecutive frustrated emotions before break
  
  // Subject configuration
  SUBJECTS: {
    Mathematics: { icon: 'fa-calculator', color: '#4a6cf7' },
    Science: { icon: 'fa-flask', color: '#28a745' },
    Technology: { icon: 'fa-laptop-code', color: '#ffc107' },
    Engineering: { icon: 'fa-cogs', color: '#dc3545' }
  },
  
  // Grade levels
  GRADE_LEVELS: ['elementary', 'middle', 'high', 'college']
};
