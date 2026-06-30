import { state, updateState } from './state.js';
import { utils } from './utils.js';
import { api } from './api.js';

const SESSION_KEY = 'elevate_user_session';
const PUBLIC_INDEX_PATH = '/index.html';
const SCHOOL_SLUG_HINT_KEY = 'elevate_school_slug_hint';
const SCHOOL_SLUG_HINT_COOKIE = 'elevate_school_slug_hint';

function normalizeSchoolSlug(slug) {
  const normalized = String(slug || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s_-]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
  return normalized || null;
}

function readCookie(name) {
  const encoded = encodeURIComponent(String(name || ''));
  const chunks = String(document.cookie || '').split(';');
  for (const rawChunk of chunks) {
    const chunk = rawChunk.trim();
    if (!chunk.startsWith(`${encoded}=`)) continue;
    return decodeURIComponent(chunk.slice(encoded.length + 1));
  }
  return null;
}

function rememberSchoolSlugHint(slug) {
  const normalized = normalizeSchoolSlug(slug);
  if (!normalized) {
    clearSchoolSlugHint();
    return null;
  }

  sessionStorage.setItem(SCHOOL_SLUG_HINT_KEY, normalized);
  localStorage.setItem(SCHOOL_SLUG_HINT_KEY, normalized);
  document.cookie = `${SCHOOL_SLUG_HINT_COOKIE}=${encodeURIComponent(normalized)}; path=/; max-age=${60 * 60 * 24 * 30}; samesite=lax`;
  return normalized;
}

function clearSchoolSlugHint() {
  sessionStorage.removeItem(SCHOOL_SLUG_HINT_KEY);
  localStorage.removeItem(SCHOOL_SLUG_HINT_KEY);
  document.cookie = `${SCHOOL_SLUG_HINT_COOKIE}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax`;
}

function getSchoolSlugHint() {
  return (
    normalizeSchoolSlug(sessionStorage.getItem(SCHOOL_SLUG_HINT_KEY))
    || normalizeSchoolSlug(localStorage.getItem(SCHOOL_SLUG_HINT_KEY))
    || normalizeSchoolSlug(readCookie(SCHOOL_SLUG_HINT_COOKIE))
  );
}

export const auth = {
  // ------- Session helpers (supports both localStorage and sessionStorage) -------

  loadSession() {
    try {
      // Try localStorage first (Remember Me = checked)
      let raw = localStorage.getItem(SESSION_KEY);
      let storage = 'localStorage';
      
      // If not in localStorage, try sessionStorage (Remember Me = unchecked)
      if (!raw) {
        raw = sessionStorage.getItem(SESSION_KEY);
        storage = 'sessionStorage';
      }
      
      if (!raw) return null;
      
      const session = JSON.parse(raw);
      if (session && session.user && session.token) {
        console.log(`✅ Session loaded from ${storage}`);
        rememberSchoolSlugHint(session.user.school_slug);
        updateState({ currentUser: session.user });
        return session;
      }
    } catch (e) {
      console.warn('Failed to parse session', e);
      localStorage.removeItem(SESSION_KEY);
      sessionStorage.removeItem(SESSION_KEY);
    }
    return null;
  },

  saveSession(user, token, rememberMe = false) {
    const session = { user, token };
    rememberSchoolSlugHint(user?.school_slug);
    
    if (rememberMe) {
      // Remember Me checked: use localStorage (persists across browser restarts)
      localStorage.setItem(SESSION_KEY, JSON.stringify(session));
      sessionStorage.removeItem(SESSION_KEY); // Clean up sessionStorage
      console.log('💾 Session saved to localStorage (Remember Me: ON)');
    } else {
      // Remember Me unchecked: use sessionStorage (only during browser session)
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
      localStorage.removeItem(SESSION_KEY); // Clean up localStorage
      console.log('⏱️ Session saved to sessionStorage (Remember Me: OFF)');
    }
    
    updateState({ currentUser: user });
  },

  clearSession() {
    localStorage.removeItem(SESSION_KEY);
    sessionStorage.removeItem(SESSION_KEY);
    clearSchoolSlugHint();
    updateState({ currentUser: null });
  },

  // ------- Auth flows (via backend API) -------

  async login(email, password, rememberMe = false) {
    try {
      if (!email || !password) {
        return { success: false, error: 'Please enter both email and password' };
      }

      const result = await api.auth.login(email, password, getSchoolSlugHint());
      console.log('Login result:', { hasUser: !!result.user, hasToken: !!result.token, rememberMe });
      this.saveSession(result.user, result.token, rememberMe);
      
      // FIX: Return the user object so the login page knows the school_slug!
      return { success: true, user: result.user }; 
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        error: error?.userMessage || error?.message || utils.getMessage('auth.login_failed'),
      };
    }
  },

  async signup(nameOrPayload, email, password, grade, role = 'student') {
    try {
      const payload = (typeof nameOrPayload === 'object' && nameOrPayload !== null)
        ? {
          ...nameOrPayload,
          role: String(nameOrPayload.role || 'student').toLowerCase(),
        }
        : {
          name: nameOrPayload,
          email,
          password,
          grade,
          role: String(role || 'student').toLowerCase(),
        };

      if (!payload.name || !payload.email || !payload.password) {
        return { success: false, error: 'All fields are required' };
      }

      if (payload.role !== 'admin' && !payload.grade) {
        return { success: false, error: 'Grade level is required for student and teacher accounts' };
      }

      if (payload.role === 'admin' && !payload.school_name) {
        return { success: false, error: 'School name is required for admin account setup' };
      }
      if (payload.role === 'admin' && !payload.school_slug) {
        return { success: false, error: 'School slug is required for admin account setup' };
      }

      // Basic client-side validation
      if (String(payload.password).length < 8) {
        return { success: false, error: 'Password must be at least 8 characters long' };
      }

      const result = await api.auth.signup(payload);
      this.saveSession(result.user, result.token);
      return { success: true };
    } catch (error) {
      console.error('Signup error:', error);

      return {
        success: false,
        error: error?.userMessage || error?.message || utils.getMessage('auth.signup_failed'),
      };
    }
  },

  // Logout
  async logout() {
    try {
      await api.auth.logout();
    } catch (error) {
      console.warn('Logout API call failed:', error);
    }
    
    this.clearSession();
    // Replace current page (dashboard/learning/etc) with login so back button
    // does not re-open protected pages from this session
    utils.navigateTo(PUBLIC_INDEX_PATH, true);
  },

  // Check Auth
  requireAuth() {
    const session = this.loadSession();
    if (session) {
      return true;
    }
    console.warn('No session found, redirecting to login.');
    // Use replace to avoid a broken page in the back-stack
    window.location.replace(PUBLIC_INDEX_PATH);
    return false;
  }
};