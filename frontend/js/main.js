// Main entry point
import { config } from './config.js';
import { state, updateState } from './state.js';
import { utils } from './utils.js';
import { auth } from './auth.js';
import { questionManager } from './question-manager.js';
import { emotionDetector } from './emotion-detector-tfjs.js';
import { adaptiveLearning } from './adaptive-learning.js';
import { progressTracker } from './progress-tracker.js';
import { storage } from './storage.js';

console.log('Elevate: main.js loaded successfully');

const SESSION_KEY = 'elevate_user_session';
const SCHOOL_SLUG_HINT_KEY = 'elevate_school_slug_hint';
const SCHOOL_SLUG_HINT_COOKIE = 'elevate_school_slug_hint';
let currentShellRoute = null;

const STUDENT_SHELL_ROUTE_TO_FILE = {
  dashboard: 'dashboard.html',
  learning: 'learning.html',
  reports: 'reports.html',
  settings: 'settings.html',
  profile: 'profile.html'
};

function normalizeUserRole(role) {
  const normalized = String(role || 'student').trim().toLowerCase();
  return ['student', 'teacher', 'admin'].includes(normalized) ? normalized : 'student';
}

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
    sessionStorage.removeItem(SCHOOL_SLUG_HINT_KEY);
    localStorage.removeItem(SCHOOL_SLUG_HINT_KEY);
    document.cookie = `${SCHOOL_SLUG_HINT_COOKIE}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax`;
    return null;
  }

  sessionStorage.setItem(SCHOOL_SLUG_HINT_KEY, normalized);
  localStorage.setItem(SCHOOL_SLUG_HINT_KEY, normalized);
  document.cookie = `${SCHOOL_SLUG_HINT_COOKIE}=${encodeURIComponent(normalized)}; path=/; max-age=${60 * 60 * 24 * 30}; samesite=lax`;
  return normalized;
}

function getStoredSchoolSlugHint() {
  return (
    normalizeSchoolSlug(sessionStorage.getItem(SCHOOL_SLUG_HINT_KEY))
    || normalizeSchoolSlug(localStorage.getItem(SCHOOL_SLUG_HINT_KEY))
    || normalizeSchoolSlug(readCookie(SCHOOL_SLUG_HINT_COOKIE))
  );
}

function getCurrentPathSlug() {
  // 1. Safely check URL parameters first (Best for static sites)
  const params = new URLSearchParams(window.location.search);
  if (params.has('school')) {
    return normalizeSchoolSlug(params.get('school'));
  }
  
  // 2. Fallback to path segment (Legacy support)
  const segments = String(window.location.pathname || '').split('/').filter(Boolean);
  if (segments.length === 1) {
    const onlySegment = segments[0] || '';
    if (!onlySegment || onlySegment.includes('.')) return null;
    if (onlySegment.toLowerCase() === 'api') return null;
    return normalizeSchoolSlug(onlySegment);
  }
  return null;
}

function getRoleHomePage(role, schoolSlug = null) {
  const normalizedRole = normalizeUserRole(role);
  const page = normalizedRole === 'teacher'
    ? 'teacher-dashboard.html'
    : normalizedRole === 'admin'
      ? 'admin.html'
      : 'dashboard.html';

  // FIX: Use query parameters instead of path segments so Render doesn't throw a 404
  if (schoolSlug && normalizedRole !== 'admin') {
      return `/${page}?school=${schoolSlug}`;
  }

  return `/${page}`;
}

function getScopedPagePath(pageFile) {
  const normalizedFile = String(pageFile || '').replace(/^\/+/, '');
  return `/${normalizedFile}`;
}

async function withTimeout(promise, timeoutMs, timeoutLabel = 'Operation timed out') {
  let timer = null;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(timeoutLabel)), timeoutMs);
      })
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function ensureStudentRoleSession() {
  if (!auth.requireAuth()) return null;

  const session = auth.loadSession();
  rememberSchoolSlugHint(session?.user?.school_slug);
  const role = normalizeUserRole(session?.user?.role);
  if (role !== 'student') {
    window.location.replace(getRoleHomePage(role, session?.user?.school_slug));
    return null;
  }

  return session;
}

function getShellRouteFromHash() {
  const hash = (window.location.hash || '').replace('#', '').trim();
  return STUDENT_SHELL_ROUTE_TO_FILE[hash] ? hash : 'dashboard';
}

function routeFromStandalonePage(page) {
  if (page === 'learning.html') return 'learning';
  if (page === 'reports.html') return 'reports';
  if (page === 'settings.html') return 'settings';
  if (page === 'profile.html') return 'profile';
  return null;
}

function redirectStandaloneStudentPageToShell(page) {
  const route = routeFromStandalonePage(page);
  if (!route) return false;
  // Preserve the ?school=slug parameter when redirecting
  window.location.replace(`${getScopedPagePath('dashboard.html')}${window.location.search}#${route}`);
  return true;
}

function setActiveSidebarRoute(route) {
  const links = document.querySelectorAll('.sidebar .nav-link[data-shell-route]');
  links.forEach(link => {
    link.classList.toggle('active', link.getAttribute('data-shell-route') === route);
  });
}

async function loadStudentShellRoute(route, options = {}) {
  const { pushState = true, replaceState = false } = options;
  if (!STUDENT_SHELL_ROUTE_TO_FILE[route]) route = 'dashboard';

  if (currentShellRoute === 'learning' && route !== 'learning') {
    // Keep camera stream alive while leaving learning, but stop active loops/timers.
    stopQuestionTimer();
    emotionDetector.prepareForRouteChange();
  }

  if (route !== 'learning' && learningReadinessWatcher) {
    clearInterval(learningReadinessWatcher);
    learningReadinessWatcher = null;
  }

  const contentArea = document.querySelector('.content-area');
  if (!contentArea) return;

  const isFirstDashboard = route === 'dashboard' && currentShellRoute === null && document.getElementById('progressModule');
  if (!isFirstDashboard) {
    const pagePath = getScopedPagePath(STUDENT_SHELL_ROUTE_TO_FILE[route]);
    const html = await withTimeout(
      fetch(pagePath, { cache: 'no-store' }).then(r => r.text()),
      8000,
      `Route content load timed out for ${pagePath}`
    );
    const parsed = new DOMParser().parseFromString(html, 'text/html');
    const nextContentArea = parsed.querySelector('.content-area');
    if (!nextContentArea) return;
    contentArea.innerHTML = nextContentArea.innerHTML;
  }

  currentShellRoute = route;
  setActiveSidebarRoute(route);

  // FIX: Preserve the ?school=slug parameter during SPA navigation
  const currentSearch = window.location.search;
  const shellUrl = `${getScopedPagePath('dashboard.html')}${currentSearch}#${route}`;
  
  if (replaceState) {
    history.replaceState({ shellRoute: route }, '', shellUrl);
  } else if (pushState) {
    history.pushState({ shellRoute: route }, '', shellUrl);
  }

  if (route === 'dashboard') {
    await loadDashboardStats();
    return;
  }
  if (route === 'learning') {
    await initLearningPage();
    return;
  }
  if (route === 'reports') {
    await initReportsPage();
    return;
  }
  if (route === 'settings') {
    initSettingsPage();
    return;
  }
  if (route === 'profile') {
    initProfilePage();
  }
}

async function setupStudentShellRouter() {
  const page = resolveCurrentFrontendPage(window.location.pathname);
  if (page !== 'dashboard.html') return false;

  window.__elevateShellRouting = true;

  const sidebarLinks = document.querySelectorAll('.sidebar .nav-link[href$=".html"]');
  sidebarLinks.forEach(link => {
    const href = link.getAttribute('href');
    const route = href.replace('.html', '');
    if (!STUDENT_SHELL_ROUTE_TO_FILE[route]) return;
    link.setAttribute('data-shell-route', route);
    link.setAttribute('href', `#${route}`);
    link.addEventListener('click', async (e) => {
      e.preventDefault();
      if (currentShellRoute === route) return;
      await loadStudentShellRoute(route, { pushState: true });
    });
  });

  window.addEventListener('popstate', async (event) => {
    if (!window.__elevateShellRouting) return;
    const route = (event.state && event.state.shellRoute) || getShellRouteFromHash();
    await loadStudentShellRoute(route, { pushState: false });
  });

  const initialRoute = getShellRouteFromHash();
  await loadStudentShellRoute(initialRoute, { pushState: false, replaceState: true });
  return true;
}

function resolveCurrentFrontendPage(pathname) {
  const segments = String(pathname || '').split('/').filter(Boolean);
  if (segments.length === 0) return 'index.html';

  const last = String(segments[segments.length - 1] || '').toLowerCase();
  if (!last) return 'index.html';

  if (last.endsWith('.html')) return last;
  if (last === 'frontend') return 'index.html';

  // Slug-root route (e.g., /nhitm) serves index.html and should initialize auth.
  if (segments.length === 1) return 'index.html';

  return last;
}

// Hydrate navbar user info ASAP from storage to avoid visible "Student" flicker
// during page-to-page navigation.
function hydrateNavbarUserFromStorage() {
  const userInfoEl = document.querySelector('.user-info');
  if (!userInfoEl) return;

  let session = null;
  try {
    const raw = localStorage.getItem(SESSION_KEY) || sessionStorage.getItem(SESSION_KEY);
    session = raw ? JSON.parse(raw) : null;
  } catch (error) {
    console.warn('Unable to hydrate navbar from storage:', error);
  }

  if (session && session.user) {
    const nameEl = document.getElementById('userName');
    const avatarEl = document.getElementById('userAvatar');
    if (nameEl) nameEl.textContent = session.user.name || 'Student';
    if (avatarEl) avatarEl.textContent = (session.user.name || 'S').charAt(0).toUpperCase();
  }

  userInfoEl.classList.add('hydrated');
}

// Initialize app based on current page
document.addEventListener('DOMContentLoaded', () => {
  hydrateNavbarUserFromStorage();

  // Clear navigation flag on successful page load
  sessionStorage.removeItem('navigating');
  
  const path = window.location.pathname;
  const page = resolveCurrentFrontendPage(path);
  
  console.log('Current Page Detected:', page || 'index (root)');

  // Handle Root/Index
  if (page === 'index.html' || page === '' || page === 'frontend') {
    initAuthPage();
  } 
  else if (page === 'dashboard.html') {
    initDashboard();
  } 
  else if (page === 'learning.html') {
    if (!redirectStandaloneStudentPageToShell(page)) initLearningPage();
  } 
  else if (page === 'reports.html') {
    if (!redirectStandaloneStudentPageToShell(page)) initReportsPage();
  } 
  else if (page === 'profile.html') {
    if (!redirectStandaloneStudentPageToShell(page)) initProfilePage();
  }
  else if (page === 'settings.html') {
    if (!redirectStandaloneStudentPageToShell(page)) initSettingsPage();
  }
  else {
    const hasAuthShell = Boolean(document.getElementById('authContainer'));
    if (hasAuthShell) {
      console.warn('Unknown path served with auth shell; initializing auth fallback for page:', page);
      initAuthPage();
    }
  }
});

// ===== SESSION CHECK FUNCTION =====
function checkExistingSession() {
  const loader = document.getElementById('sessionCheckLoader');
  const authContainer = document.getElementById('authContainer');

  const revealAuth = () => {
    if (loader) loader.style.display = 'none';
    if (authContainer) authContainer.style.display = 'block';
  };

  // Safety net: never leave the page on an infinite loading state.
  const loaderFailsafe = setTimeout(() => {
    const authHidden = authContainer && authContainer.style.display === 'none';
    const loaderVisible = loader && loader.style.display !== 'none';
    if (loaderVisible && authHidden) {
      console.warn('Session check timeout fallback triggered; revealing auth container.');
      revealAuth();
    }
  }, 2500);
  
  console.log('🔍 Checking for existing session...');
  
  // Try to load existing session
  const session = auth.loadSession();
  
  setTimeout(() => {
    try {
      if (session && session.user && session.token) {
        console.log('✅ Valid session found! User:', session.user.name);
        console.log('🚀 Auto-redirecting to dashboard...');

        // Session exists, redirect based on role
        const role = normalizeUserRole(session.user.role);
        rememberSchoolSlugHint(session.user.school_slug);
        const dest = getRoleHomePage(role, session.user.school_slug);
        clearTimeout(loaderFailsafe);
        window.location.replace(dest);
      } else {
        console.log('❌ No valid session found. Showing login page...');

        // No session, show login form
        clearTimeout(loaderFailsafe);
        revealAuth();
      }
    } catch (error) {
      console.error('Session bootstrap failed; falling back to auth form.', error);
      clearTimeout(loaderFailsafe);
      revealAuth();
    }
  }, 500); // Small delay for smooth UX
}

// ===== PASSWORD VISIBILITY TOGGLE =====
function setupPasswordToggles() {
  const toggleIcons = document.querySelectorAll('.password-toggle');
  
  toggleIcons.forEach(icon => {
    icon.addEventListener('click', function() {
      const targetId = this.getAttribute('data-target');
      const passwordInput = document.getElementById(targetId);
      
      if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        this.classList.remove('fa-eye');
        this.classList.add('fa-eye-slash');
      } else {
        passwordInput.type = 'password';
        this.classList.remove('fa-eye-slash');
        this.classList.add('fa-eye');
      }
    });
  });
}

// ===== AUTH PAGE (index.html) =====
function initAuthPage() {
  console.log('Initializing Auth Page...');

  rememberSchoolSlugHint(getStoredSchoolSlugHint() || getCurrentPathSlug());

  const canonicalAuthPath = '/index.html';
  const currentPath = String(window.location.pathname || '');
  if (currentPath !== '/' && currentPath !== canonicalAuthPath) {
    // Keep auth page URL slug-free until a successful login redirect chooses scoped route.
    window.history.replaceState({}, document.title, `${canonicalAuthPath}${window.location.search || ''}`);
  }
  
  // Check for existing session first
  checkExistingSession();
  
  // Setup password visibility toggles
  setupPasswordToggles();

  // Setup role selector buttons (shared helper)
  function setupRoleSelector(selectorId, onChange) {
    const container = document.getElementById(selectorId);
    if (!container) return;
    container.querySelectorAll('.role-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        container.querySelectorAll('.role-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const selectedRole = btn.getAttribute('data-role') || 'student';
        if (typeof onChange === 'function') {
          onChange(selectedRole);
        }
      });
    });
  }

  function getSelectedRole(selectorId) {
    const container = document.getElementById(selectorId);
    if (!container) return 'student';
    const active = container.querySelector('.role-btn.active');
    return active ? active.getAttribute('data-role') : 'student';
  }

  function showAuthAlert(alertId, message, type = 'error') {
    const el = document.getElementById(alertId);
    if (!el) return;
    el.className = `auth-alert auth-alert-${type}`;
    el.textContent = message;
    el.style.display = 'block';
  }

  function hideAuthAlert(alertId) {
    const el = document.getElementById(alertId);
    if (el) el.style.display = 'none';
  }

  setupRoleSelector('loginRoleSelector');

  const signupStep1 = document.getElementById('signupStep1');
  const signupStep2 = document.getElementById('signupStep2');
  const signupNextBtn = document.getElementById('signupNextBtn');
  const signupBackBtn = document.getElementById('signupBackBtn');
  const signupSubmitBtn = document.getElementById('signupSubmitBtn');
  const signupRoleHint = document.getElementById('signupRoleHint');
  const signupGradeGroup = document.getElementById('signupGradeGroup');
  const signupGradeInput = document.getElementById('signupGrade');
  const signupAdminFields = document.getElementById('signupAdminFields');
  const signupSchoolNameInput = document.getElementById('signupSchoolName');
  const signupSchoolSlugInput = document.getElementById('signupSchoolSlug');
  const signupStepper = document.getElementById('signupStepper');
  let signupCurrentStep = 1;

  function setSignupStep(stepNumber) {
    signupCurrentStep = stepNumber === 2 ? 2 : 1;
    if (signupStep1) signupStep1.style.display = signupCurrentStep === 1 ? 'block' : 'none';
    if (signupStep2) signupStep2.style.display = signupCurrentStep === 2 ? 'block' : 'none';

    if (signupStepper) {
      signupStepper.querySelectorAll('.signup-step').forEach(stepNode => {
        const step = Number(stepNode.getAttribute('data-step') || 1);
        stepNode.classList.toggle('active', step === signupCurrentStep);
      });
    }
  }

  function syncSignupRoleUI() {
    const role = getSelectedRole('signupRoleSelector');
    const isAdmin = role === 'admin';

    if (signupGradeGroup) signupGradeGroup.style.display = isAdmin ? 'none' : '';
    if (signupAdminFields) signupAdminFields.style.display = isAdmin ? '' : 'none';

    if (signupGradeInput) {
      signupGradeInput.disabled = isAdmin;
      signupGradeInput.required = !isAdmin;
      if (isAdmin) signupGradeInput.value = '';
    }

    if (signupSchoolNameInput) {
      signupSchoolNameInput.disabled = !isAdmin;
      signupSchoolNameInput.required = isAdmin;
      if (!isAdmin) signupSchoolNameInput.value = '';
    }

    if (signupSchoolSlugInput) {
      signupSchoolSlugInput.disabled = !isAdmin;
      signupSchoolSlugInput.required = isAdmin;
      if (!isAdmin) signupSchoolSlugInput.value = '';
      const slugFieldWrap = signupSchoolSlugInput.closest('.mb-3');
      if (slugFieldWrap) slugFieldWrap.style.display = isAdmin ? '' : 'none';
    }

    if (signupRoleHint) {
      signupRoleHint.textContent = isAdmin
        ? 'Admin setup skips grade and collects school profile details in Step 2.'
        : 'Student/Teacher setup includes grade level in Step 2.';
    }
  }

  function validateSignupStepOne() {
    const name = document.getElementById('signupName')?.value?.trim() || '';
    const email = document.getElementById('signupEmail')?.value?.trim() || '';
    const password = document.getElementById('signupPassword')?.value || '';

    if (!name || !email || !password) {
      showAuthAlert('signupAlert', 'Please complete name, email, and password before continuing.', 'warning');
      return false;
    }

    if (password.length < 8) {
      showAuthAlert('signupAlert', 'Password must be at least 8 characters long.', 'warning');
      return false;
    }

    return true;
  }

  function resetSignupFlow() {
    setSignupStep(1);
    hideAuthAlert('signupAlert');
    syncSignupRoleUI();
  }

  setupRoleSelector('signupRoleSelector', () => {
    syncSignupRoleUI();
    hideAuthAlert('signupAlert');
  });
  syncSignupRoleUI();
  setSignupStep(1);

  if (signupNextBtn) {
    signupNextBtn.addEventListener('click', () => {
      hideAuthAlert('signupAlert');
      if (!validateSignupStepOne()) return;
      setSignupStep(2);
    });
  }

  if (signupBackBtn) {
    signupBackBtn.addEventListener('click', () => {
      hideAuthAlert('signupAlert');
      setSignupStep(1);
    });
  }
  
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');
  const showSignupLink = document.getElementById('showSignup');
  const showLoginLink = document.getElementById('showLogin');

  // Toggle Forms
  if (showSignupLink) {
    showSignupLink.addEventListener('click', (e) => {
      e.preventDefault();
      loginForm.style.display = 'none';
      signupForm.style.display = 'block';
      hideAuthAlert('loginAlert');
      resetSignupFlow();
    });
  }
  
  if (showLoginLink) {
    showLoginLink.addEventListener('click', (e) => {
      e.preventDefault();
      signupForm.style.display = 'none';
      loginForm.style.display = 'block';
      hideAuthAlert('signupAlert');
      setSignupStep(1);
    });
  }

  // Handle Login
  if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
      e.preventDefault(); // Stop page reload
      console.log('Login form submitted');

      const email = document.getElementById('loginEmail').value;
      const password = document.getElementById('loginPassword').value;
      const rememberMe = document.getElementById('rememberMe').checked;
      const btn = loginForm.querySelector('button');
      
      console.log('Remember Me:', rememberMe ? 'CHECKED ✓' : 'UNCHECKED ✗');
      
      const originalText = btn.innerText;
      btn.innerText = 'Signing in...';
      btn.disabled = true;

      const result = await auth.login(email, password, rememberMe);
      
      if (result.success) {
        console.log('Login success, navigating by role...');
        const session = auth.loadSession();
        const role = normalizeUserRole(session && session.user ? session.user.role : 'student');
        const selectedRole = getSelectedRole('loginRoleSelector');
        const dest = getRoleHomePage(role, session?.user?.school_slug);

        showAuthAlert('loginAlert', utils.getMessage('auth.login_success'), 'success');

        if (selectedRole && selectedRole !== role) {
          const roleLabel = role.charAt(0).toUpperCase() + role.slice(1);
          showAuthAlert('loginAlert', `Signed in successfully. Your account is registered as ${roleLabel}. Redirecting...`, 'info');
          setTimeout(() => utils.navigateTo(dest, true), 3500);
        } else {
          setTimeout(() => utils.navigateTo(dest, true), 900);
        }
      } else {
        console.error('Login error:', result.error);
        showAuthAlert('loginAlert', result.error || utils.getMessage('auth.login_failed'), 'error');
        btn.innerText = originalText;
        btn.disabled = false;
      }
    });
  }

  // Handle Signup
  if (signupForm) {
    signupForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (signupCurrentStep === 1) {
        if (validateSignupStepOne()) {
          setSignupStep(2);
        }
        return;
      }

      const name = document.getElementById('signupName').value.trim();
      const email = document.getElementById('signupEmail').value.trim();
      const password = document.getElementById('signupPassword').value;
      const role = getSelectedRole('signupRoleSelector');
      const grade = role === 'admin' ? null : (document.getElementById('signupGrade').value || null);
      const schoolName = role === 'admin' ? (document.getElementById('signupSchoolName')?.value?.trim() || '') : '';
      const schoolSlug = role === 'admin' ? normalizeSchoolSlug(document.getElementById('signupSchoolSlug')?.value) : null;

      if (role !== 'admin' && !grade) {
        showAuthAlert('signupAlert', 'Please select a grade level to continue.', 'warning');
        return;
      }

      if (role === 'admin' && !schoolName) {
        showAuthAlert('signupAlert', 'Please enter a school name for the admin workspace.', 'warning');
        return;
      }

      if (role === 'admin' && !schoolSlug) {
        showAuthAlert('signupAlert', 'Please enter a valid school slug for the admin workspace.', 'warning');
        return;
      }

      const btn = signupSubmitBtn || signupForm.querySelector('button[type="submit"]');
      
      const originalText = btn.innerText;
      btn.innerText = 'Creating account...';
      btn.disabled = true;

      const signupPayload = {
        name,
        email,
        password,
        role,
      };

      if (grade) signupPayload.grade = grade;
      if (role === 'admin') {
        signupPayload.school_name = schoolName;
        signupPayload.school_slug = schoolSlug;
      }
      
      const result = await auth.signup(signupPayload);
      
      if (result.success) {
        showAuthAlert('signupAlert', utils.getMessage('auth.signup_success'), 'success');
        const session = auth.loadSession();
        const savedRole = normalizeUserRole((session && session.user ? session.user.role : null) || role);
        const dest = getRoleHomePage(savedRole, session?.user?.school_slug);
        setTimeout(() => utils.navigateTo(dest, true), 1200);
      } else {
        console.error('Signup error:', result.error);
        showAuthAlert('signupAlert', result.error || utils.getMessage('auth.signup_failed'), 'error');
        btn.innerText = originalText;
        btn.disabled = false;
      }
    });
  }
}

// ===== DASHBOARD =====
const DASHBOARD_STATS_CACHE_KEY = 'elevate_dashboard_stats_v1';

async function initDashboard() {
  if (!ensureStudentRoleSession()) return;

  console.log('Initializing Dashboard...');

  try {
    updateUserInfo();
    setupProfileMenu();
    setupLogout();

    const shellReady = await withTimeout(
      setupStudentShellRouter(),
      9000,
      'Dashboard shell initialization timed out'
    );

    if (shellReady) {
      return;
    }

    installBackNavigationGuard();

    // Clear cached stats when the tab/window is closed so the skeleton
    // shows fresh on the next browser session.
    window.addEventListener('beforeunload', () => {
      sessionStorage.removeItem(DASHBOARD_STATS_CACHE_KEY);
    });

    await withTimeout(loadDashboardStats(), 10000, 'Dashboard stats load timed out');
  } catch (error) {
    console.error('Dashboard bootstrap failed:', error);
    showDashboardPanel('empty');
    utils.showNotification('Dashboard is taking too long. Showing fallback view.', 'warning');
  }
}

// Apply raw stats object to the DOM elements.
function applyDashboardStats(stats) {
  const totalQuestionsEl = document.getElementById('dashboardTotalQuestions');
  const accuracyEl       = document.getElementById('dashboardAccuracy');
  const streakEl         = document.getElementById('dashboardStreak');
  const levelEl          = document.getElementById('dashboardLevel');

  if (totalQuestionsEl) totalQuestionsEl.textContent = stats.totalQuestions || 0;
  if (accuracyEl)        accuracyEl.textContent       = `${stats.accuracy || 0}%`;
  if (streakEl)          streakEl.textContent         = stats.streak || 0;
  if (levelEl)           levelEl.textContent          = stats.level || 'Beginner';
}

// Show the right panel (skeleton / stats / empty) cleanly.
function showDashboardPanel(panel) {
  const loadingEl    = document.getElementById('dashboardLoading');
  const statsEl      = document.getElementById('progressStats');
  const emptyEl      = document.getElementById('progressEmptyState');

  if (loadingEl) loadingEl.style.display = panel === 'loading' ? 'block' : 'none';
  if (statsEl)   statsEl.style.display   = panel === 'stats'   ? 'flex'  : 'none';
  if (emptyEl)   emptyEl.style.display   = panel === 'empty'   ? 'flex'  : 'none';
}

async function loadDashboardStats() {
  // ── 1. Try to serve from sessionStorage cache first ──────────
  const raw = sessionStorage.getItem(DASHBOARD_STATS_CACHE_KEY);
  if (raw) {
    try {
      const cached = JSON.parse(raw);
      applyDashboardStats(cached);
      showDashboardPanel(cached.totalQuestions > 0 ? 'stats' : 'empty');

      // Silently refresh in the background — no skeleton flash
      refreshDashboardStatsInBackground();
      return;
    } catch (_) {
      sessionStorage.removeItem(DASHBOARD_STATS_CACHE_KEY);
    }
  }

  // ── 2. No cache — show skeleton and wait for the fetch ───────
  showDashboardPanel('loading');

  try {
    const { api } = await import('./api.js');
    const stats = await api.progress.getDashboardStats();

    applyDashboardStats(stats);
    sessionStorage.setItem(DASHBOARD_STATS_CACHE_KEY, JSON.stringify(stats));

    showDashboardPanel(stats.totalQuestions > 0 ? 'stats' : 'empty');
    console.log('Dashboard stats loaded:', stats);
  } catch (error) {
    // On error still hide the skeleton — show empty state as a fallback
    console.error('Failed to load dashboard stats:', error);
    showDashboardPanel('empty');
    utils.showNotification('Failed to load statistics', 'error');
  }
}

// Background refresh that updates numbers without any visible flash.
async function refreshDashboardStatsInBackground() {
  try {
    const { api } = await import('./api.js');
    const stats = await api.progress.getDashboardStats();

    applyDashboardStats(stats);
    sessionStorage.setItem(DASHBOARD_STATS_CACHE_KEY, JSON.stringify(stats));

    // If DB now has data but we were showing empty, switch to stats.
    const statsEl = document.getElementById('progressStats');
    const emptyEl = document.getElementById('progressEmptyState');
    if (stats.totalQuestions > 0 && statsEl && statsEl.style.display === 'none') {
      showDashboardPanel('stats');
    }
    console.log('Dashboard stats silently refreshed:', stats);
  } catch (error) {
    // Non-critical — cached data is already displayed
    console.warn('Background refresh failed:', error);
  }
}

// ===== LEARNING =====
const STEM_SUBJECTS = [
  { slug: 'science', label: 'Science', icon: 'fa-flask' },
  { slug: 'technology', label: 'Technology', icon: 'fa-laptop-code' },
  { slug: 'engineering', label: 'Engineering', icon: 'fa-cogs' },
  { slug: 'mathematics', label: 'Mathematics', icon: 'fa-calculator' }
];

const STEM_TOPICS_FALLBACK = {
  science: ['physics', 'chemistry', 'biology', 'earth_science', 'environmental_science'],
  technology: ['computer_fundamentals', 'programming', 'ai_basics', 'networks', 'internet_safety'],
  engineering: ['design_thinking', 'structures', 'mechanics', 'electrical_basics', 'robotics'],
  mathematics: ['arithmetic', 'algebra', 'geometry', 'statistics', 'calculus']
};

function normalizeTopicSlug(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/-/g, '_')
    .replace(/\s+/g, '_');
}

function formatTopicLabel(slug) {
  const words = normalizeTopicSlug(slug)
    .split('_')
    .filter(Boolean);
  return words.map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

let learningReadinessWatcher = null;
let learningVisibilityHandlerAttached = false;
let assignedLearningTests = [];
let selectedAssignedLearning = null;
let activeAssignedTestSession = null;
let learningMode = 'practice';

function getSelectedLearningMode() {
  if (isAssignedLearningMode()) return 'assigned';
  return learningMode === 'assigned' ? 'assigned' : 'practice';
}

function setLearningMode(mode, options = {}) {
  const { preserveAssignedSelection = true } = options;
  learningMode = mode === 'assigned' ? 'assigned' : 'practice';

  if (learningMode === 'practice' && !preserveAssignedSelection) {
    clearAssignedLearningMode();
  }

  const radios = document.querySelectorAll('input[name="learningMode"]');
  radios.forEach(radio => {
    radio.checked = radio.value === learningMode;
  });

  updateLearningModeUI();
  updateLearningApplyButtonState();
}

function updateLearningModeUI() {
  const assignedSection = document.getElementById('assignedModeSection');
  const practiceSection = document.getElementById('practiceModeSection');
  const isAssignedMode = getSelectedLearningMode() === 'assigned';

  if (assignedSection) assignedSection.style.display = isAssignedMode ? '' : 'none';
  if (practiceSection) practiceSection.style.display = isAssignedMode ? 'none' : '';
}

function isAssignedLearningMode() {
  return Boolean(selectedAssignedLearning && selectedAssignedLearning.test);
}

function clearAssignedLearningMode() {
  selectedAssignedLearning = null;
  activeAssignedTestSession = null;
  const assignedSelect = document.getElementById('assignedTestSelect');
  if (assignedSelect) {
    assignedSelect.value = '';
  }
}

function getAssignedLearningLabel(item) {
  if (!item || !item.test) return 'Assigned Test';
  const test = item.test;
  const due = item.due_at ? ` | Due: ${new Date(item.due_at).toLocaleDateString()}` : '';
  return `${test.title} (${test.subject}, ${utils.getGradeDisplayName(test.grade)})${due}`;
}

async function loadAssignedLearningTests() {
  try {
    const { api } = await import('./api.js');
    const response = await api.student.getAssignedTests();
    assignedLearningTests = Array.isArray(response.assignments) ? response.assignments : [];

    const select = document.getElementById('assignedTestSelect');
    if (!select) return;

    const rows = ['<option value="">Select an assigned test</option>'];
    assignedLearningTests.forEach(item => {
      if (!item || !item.test) return;
      const disabled = item.already_taken || item.is_overdue;
      const statusSuffix = item.already_taken ? ' [Completed]' : (item.is_overdue ? ' [Overdue]' : '');
      rows.push(`<option value="${item.id}" ${disabled ? 'disabled' : ''}>${getAssignedLearningLabel(item)}${statusSuffix}</option>`);
    });
    select.innerHTML = rows.join('');

    if (selectedAssignedLearning) {
      const stillExists = assignedLearningTests.find(item => Number(item.id) === Number(selectedAssignedLearning.id));
      if (!stillExists) {
        selectedAssignedLearning = null;
        activeAssignedTestSession = null;
      }
    }
  } catch (error) {
    console.warn('Failed to load assigned tests for learning:', error);
  }
}

function getSubjectLabelBySlug(slug) {
  const item = STEM_SUBJECTS.find(s => s.slug === slug);
  return item ? item.label : slug;
}

function getApiSubjectValues(slug) {
  const mapping = {
    science: ['Science', 'science', 'Physics', 'physics', 'Chemistry', 'chemistry', 'Biology', 'biology'],
    technology: ['Technology', 'technology', 'Computer Science', 'computer science', 'Programming', 'programming'],
    engineering: ['Engineering', 'engineering'],
    mathematics: ['Mathematics', 'mathematics', 'Math', 'math']
  };
  return mapping[slug] || [slug];
}

function isCameraRequiredForLearning() {
  if (getSelectedLearningMode() === 'assigned') {
    return true;
  }
  const settings = getEffectiveSettings();
  return settings.requireCamera !== false;
}

function hasLiveCameraStream() {
  const stream = state.cameraStream;
  if (!stream || typeof stream.getTracks !== 'function') return false;
  const tracks = stream.getVideoTracks();
  return tracks.some(track => track.readyState === 'live');
}

function isLearningControlReady() {
  if (!isCameraRequiredForLearning()) {
    return true;
  }
  return state.cameraActive && hasLiveCameraStream() && state.modelsLoaded && !state.usingSimulatedEmotions && state.faceDetectionConfirmed;
}

function updateLearningStatusCard() {
  const lockBanner = document.getElementById('learningLockBanner');
  const detectionBadge = document.getElementById('detectionStateBadge');
  const emotionLive = document.getElementById('liveEmotionState');

  const ready = isLearningControlReady();
  const cameraRequired = isCameraRequiredForLearning();
  if (lockBanner) {
    lockBanner.className = `learning-lock-banner ${ready ? 'ready' : 'locked'}`;
    if (!cameraRequired) {
      lockBanner.innerHTML = '<i class="fas fa-toggle-off"></i> Camera requirement is off. You can continue learning without camera.';
    } else {
      lockBanner.innerHTML = ready
        ? '<i class="fas fa-unlock"></i> Camera and face detection are active. You can now select grade, STEM subject, and topic.'
        : '<i class="fas fa-lock"></i> Controls are locked until camera is ON and real face detection points are visible on your face.';
    }
  }

  if (detectionBadge) {
    detectionBadge.className = `status-pill ${ready ? 'on' : 'off'}`;
    detectionBadge.textContent = cameraRequired
      ? (ready ? 'Detection Confirmed' : 'Waiting for Detection')
      : 'Camera Optional';
  }

  if (emotionLive) {
    const current = state.currentEmotion || 'Neutral';
    emotionLive.textContent = String(current);
  }
}

function setLearningControlsEnabled(enabled) {
  const gradeSelect = document.getElementById('gradeSelect');
  const topicSelect = document.getElementById('topicSelect');
  const difficultySelect = document.getElementById('difficultySelect');
  const countSelect = document.getElementById('countSelect');
  const assignedSelect = document.getElementById('assignedTestSelect');
  const refreshAssignedBtn = document.getElementById('refreshAssignedTests');
  const applyBtn = document.getElementById('applySelectionBtn');
  if (gradeSelect) {
    gradeSelect.disabled = !enabled;
    gradeSelect.style.opacity = enabled ? '1' : '0.6';
  }
  if (topicSelect) {
    topicSelect.disabled = !enabled;
    topicSelect.style.opacity = enabled ? '1' : '0.6';
  }
  if (difficultySelect) {
    difficultySelect.disabled = !enabled;
    difficultySelect.style.opacity = enabled ? '1' : '0.6';
  }
  if (countSelect) {
    countSelect.disabled = !enabled;
    countSelect.style.opacity = enabled ? '1' : '0.6';
  }
  if (assignedSelect) {
    assignedSelect.disabled = !enabled;
    assignedSelect.style.opacity = enabled ? '1' : '0.6';
  }
  if (refreshAssignedBtn) {
    refreshAssignedBtn.disabled = !enabled;
    refreshAssignedBtn.style.opacity = enabled ? '1' : '0.6';
  }

  // Keep mode switching always available so users can return to Practice even
  // when Assigned mode currently requires camera/detection.
  document.querySelectorAll('input[name="learningMode"]').forEach(input => {
    input.disabled = false;
  });

  document.querySelectorAll('.subject-btn').forEach(btn => {
    btn.disabled = !enabled;
    btn.style.opacity = enabled ? '1' : '0.6';
  });

  if (applyBtn) {
    applyBtn.disabled = !enabled || !hasLearningSelectionComplete();
  }
}

function hasLearningSelectionComplete() {
  if (getSelectedLearningMode() === 'assigned') {
    return isAssignedLearningMode();
  }
  const gradeValue = state.selectedGrade || document.getElementById('gradeSelect')?.value;
  const topicValue = state.selectedTopic || document.getElementById('topicSelect')?.value;
  const difficultyValue = getSelectedPracticeDifficulty();
  const countValue = state.questionsPerSession || parseInt(document.getElementById('countSelect')?.value || '0', 10);
  return Boolean(gradeValue && state.selectedSubject && topicValue && difficultyValue && countValue);
}

function getSelectedPracticeDifficulty() {
  const allowed = new Set(['adaptive', 'easy', 'medium', 'hard']);
  const fromState = String(state.selectedPracticeDifficulty || '').toLowerCase();
  const fromControl = String(document.getElementById('difficultySelect')?.value || '').toLowerCase();
  const candidate = fromState || fromControl || 'adaptive';
  return allowed.has(candidate) ? candidate : 'adaptive';
}

function updateLearningApplyButtonState() {
  const applyBtn = document.getElementById('applySelectionBtn');
  if (!applyBtn) return;
  applyBtn.disabled = !(isLearningControlReady() && hasLearningSelectionComplete());
}

function setQuestionInteractionLocked(locked) {
  document.querySelectorAll('input[name="answer"]').forEach(input => {
    input.disabled = locked;
  });

  const submitBtn = document.getElementById('submitAnswer');
  const nextBtn = document.getElementById('nextQuestion');
  const finishBtn = document.getElementById('finishTest');
  if (submitBtn) submitBtn.disabled = locked;
  if (nextBtn) nextBtn.disabled = locked;
  if (finishBtn) finishBtn.disabled = locked;
}

function markLearningSelectionPending(showMessage = false) {
  stopQuestionTimer();
  updateState({ questions: [], currentQuestionIndex: 0, questionStartTime: null });

  const questionContainer = document.getElementById('questionContainer');
  if (questionContainer) {
    questionContainer.style.display = 'none';
  }

  if (showMessage) {
    utils.showNotification('Selection updated. Press OK to load questions.', 'info');
  }

  updateLearningApplyButtonState();
}

async function applyLearningSelection() {
  if (!isLearningControlReady()) {
    const msg = isCameraRequiredForLearning()
      ? 'Start camera and wait for real face detection before loading questions.'
      : 'Learning controls are not ready yet. Please try again.';
    utils.showNotification(msg, 'warning');
    return;
  }

  if (!hasLearningSelectionComplete()) {
    const msg = getSelectedLearningMode() === 'assigned'
      ? 'Select an assigned test, then press OK.'
      : 'Select grade, STEM subject, topic, and question count, then press OK.';
    utils.showNotification(msg, 'warning');
    return;
  }

  await loadQuestionsForCurrentSelection();
}

function ensureLearningVisibilityRecoveryHandler() {
  if (learningVisibilityHandlerAttached) return;

  document.addEventListener('visibilitychange', async () => {
    if (document.hidden) return;

    const inLearning = currentShellRoute === 'learning' || window.location.pathname.endsWith('learning.html');
    if (!inLearning) return;

    if (state.cameraActive && state.cameraStream) {
      const restored = await emotionDetector.restoreActiveSession();
      if (restored) {
        emotionDetector.startDetection();
      }
    }
  });

  learningVisibilityHandlerAttached = true;
}

function startLearningReadinessWatcher() {
  if (learningReadinessWatcher) {
    clearInterval(learningReadinessWatcher);
    learningReadinessWatcher = null;
  }

  let lastReadyState = null;
  let lastCameraRequired = null;

  const applyLearningReadinessState = () => {
    const cameraRequired = isCameraRequiredForLearning();
    const ready = isLearningControlReady();
    if (ready !== lastReadyState || cameraRequired !== lastCameraRequired) {
      setLearningControlsEnabled(ready);
      updateLearningStatusCard();
      updateLearningApplyButtonState();
      setQuestionInteractionLocked(!ready);
      if (!ready && cameraRequired) {
        stopQuestionTimer();
        const questionContainer = document.getElementById('questionContainer');
        if (questionContainer) {
          questionContainer.style.display = 'none';
        }
      }
      lastReadyState = ready;
      lastCameraRequired = cameraRequired;
      if (ready && cameraRequired) {
        utils.showNotification('Face detection confirmed. Learning controls are now enabled.', 'success');
      }
      if (!cameraRequired) {
        setQuestionInteractionLocked(false);
      }
    } else {
      updateLearningStatusCard();
      updateLearningApplyButtonState();
    }
  };

  // Apply immediately so controls do not remain stale/disabled after route change.
  applyLearningReadinessState();
  learningReadinessWatcher = setInterval(applyLearningReadinessState, 600);
}

async function initLearningPage() {
  if (!ensureStudentRoleSession()) return;
  
  updateUserInfo();
  setupProfileMenu();
  setupLogout();
  installBackNavigationGuard();
  await syncUserSettingsFromServer();
  
  // Initialize emotion detector FIRST
  const emotionInitialized = await emotionDetector.init();
  if (!emotionInitialized) {
    console.error('Emotion detector failed to initialize');
    utils.showNotification('Emotion detection unavailable', 'warning');
  }
  
  // Setup camera controls
  setupCameraControls();
  ensureLearningVisibilityRecoveryHandler();
  
  // Auto-load user's grade level from profile
  await autoLoadUserGrade();
  
  // Setup grade selector with hierarchical access
  setupGradeSelector();
  
  // Setup subject selection
  setupSubjectSelector();
  
  // Setup question handlers
  setupQuestionHandlers();
  // Setup topic selector (syllabus)
  setupTopicSelector();
  await loadAssignedLearningTests();
  applyLearningSettingsDefaults();
  setLearningMode('practice', { preserveAssignedSelection: true });

  if (state.cameraActive && state.cameraStream) {
    const restored = await emotionDetector.restoreActiveSession();
    if (restored) {
      emotionDetector.startDetection();
    }
  }
  
  // Hide question container initially until camera is active
  const questionContainer = document.getElementById('questionContainer');
  if (questionContainer) {
    questionContainer.style.display = 'none';
  }
  
  // Ensure grade and subjects are disabled on page load (camera not active yet)
  ensureControlsDisabled();
  updateLearningStatusCard();
  startLearningReadinessWatcher();
  updateLearningApplyButtonState();
  
  // Initialize subject performance widget AFTER all DOM setup
  console.log('🔄 Initializing performance widget...');
  setTimeout(() => {
    updateSubjectPerformanceWidget();
    
    // Setup refresh button for performance widget
    const refreshBtn = document.getElementById('refreshPerformanceWidget');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        const icon = refreshBtn.querySelector('i');
        if (icon) icon.classList.add('fa-spin');
        updateSubjectPerformanceWidget();
        setTimeout(() => {
          if (icon) icon.classList.remove('fa-spin');
        }, 500);
        utils.showNotification('Performance data refreshed', 'info');
      });
      console.log('✅ Refresh button initialized');
    } else {
      console.warn('❌ Refresh button not found');
    }
  }, 100);
}

function applyLearningSettingsDefaults() {
  const settings = getEffectiveSettings();
  const difficultySelect = document.getElementById('difficultySelect');
  const countSelect = document.getElementById('countSelect');

  const configuredDefaultDifficulty = String(settings.defaultDifficulty || 'adaptive').toLowerCase();
  const normalizedDifficulty = ['adaptive', 'easy', 'medium', 'hard'].includes(configuredDefaultDifficulty)
    ? configuredDefaultDifficulty
    : 'adaptive';

  if (difficultySelect) {
    difficultySelect.value = normalizedDifficulty;
    updateState({ selectedPracticeDifficulty: normalizedDifficulty });
  }

  if (!countSelect) return;

  const preferredCount = parseInt(settings.questionsPerSession, 10);
  const availableCounts = Array.from(countSelect.options || []).map(option => parseInt(option.value, 10));

  if (Number.isFinite(preferredCount) && preferredCount >= 5 && preferredCount <= 50) {
    if (!availableCounts.includes(preferredCount)) {
      const customOption = document.createElement('option');
      customOption.value = String(preferredCount);
      customOption.textContent = String(preferredCount);
      countSelect.appendChild(customOption);
    }
    countSelect.value = String(preferredCount);
    updateState({ questionsPerSession: preferredCount });
    return;
  }

  updateState({ questionsPerSession: parseInt(countSelect.value, 10) || 10 });
}

async function autoLoadUserGrade() {
  try {
    const { api } = await import('./api.js');
    const profileData = await api.auth.getProfile();
    
    if (profileData && profileData.user && profileData.user.grade) {
      const userGrade = profileData.user.grade;
      
      // Update state with user's grade and selected grade (default to user's grade)
      updateState({ 
        currentGrade: userGrade,
        selectedGrade: userGrade  // Default to user's own grade
      });
      
      console.log('🎓 Loaded user grade from profile:', userGrade);
    } else {
      console.warn('⚠️ No grade in profile, using default: middle');
      updateState({ 
        currentGrade: 'middle',
        selectedGrade: 'middle'
      });
    }
  } catch (error) {
    console.error('❌ Failed to load user grade:', error);
    console.log('🔄 Using default grade: middle');
    updateState({ 
      currentGrade: 'middle',
      selectedGrade: 'middle'
    });
  }
}

// Setup grade selector with hierarchical access
function setupGradeSelector() {
  const gradeSelect = document.getElementById('gradeSelect');
  if (!gradeSelect) return;
  
  const userGrade = state.currentGrade || 'middle';
  const accessibleGrades = utils.getAccessibleGrades(userGrade);
  
  // Populate grade dropdown with accessible grades only
  gradeSelect.innerHTML = accessibleGrades.map(grade => `
    <option value="${grade}" ${grade === userGrade ? 'selected' : ''}>
      ${utils.getGradeDisplayName(grade)}
    </option>
  `).join('');
  
  // Disabled until camera + real detection are confirmed
  gradeSelect.disabled = true;
  gradeSelect.style.opacity = '0.6';
  
  // Handle grade selection change
  gradeSelect.addEventListener('change', async (e) => {
    // Hard lock: controls require real face detection, not simulation.
    if (!isLearningControlReady()) {
      utils.showNotification('Turn on camera and wait for real face detection points to appear first.', 'warning');
      gradeSelect.value = state.selectedGrade;
      return;
    }
    
    const selectedGrade = e.target.value;
    setLearningMode('practice', { preserveAssignedSelection: false });
    updateState({ selectedGrade });
    console.log('📚 Selected grade:', utils.getGradeDisplayName(selectedGrade));

    if (state.selectedSubject) {
      updateState({ selectedTopic: null });
      await updateTopicOptions();
    }

    markLearningSelectionPending(state.selectedSubject != null);
  });
  
  console.log(`✅ Grade selector initialized - Accessible grades: ${accessibleGrades.map(g => utils.getGradeDisplayName(g)).join(', ')}`);
}

function setupSubjectSelector() {
  const subjectContainer = document.getElementById('subjectOptions');
  
  if (subjectContainer) {
    // Strict STEM subjects only
    subjectContainer.innerHTML = STEM_SUBJECTS.map(subject => `
      <button class="btn btn-outline-primary subject-btn" data-subject="${subject.slug}" title="${subject.label}">
        <i class="fas ${subject.icon}"></i> ${subject.label}
      </button>
    `).join('');
    
    // Add event listeners
    const subjectButtons = subjectContainer.querySelectorAll('.subject-btn');
    subjectButtons.forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!isLearningControlReady()) {
          utils.showNotification('Start camera and wait for face detection to unlock subjects.', 'warning');
          return;
        }

        subjectButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        setLearningMode('practice', { preserveAssignedSelection: false });
        
        const subject = btn.dataset.subject;
        updateState({ selectedSubject: subject, selectedTopic: null });
        
        console.log('\ud83d\udcda Subject selected:', subject);
        
        // Update performance widget when subject selected
        updateSubjectPerformanceWidget();
        
        // Update topic selector to reflect available topics for this subject/grade
        await updateTopicOptions();

        markLearningSelectionPending(true);
      });
    });
  }
}

// Populate topic selector based on selected subject and grade
async function updateTopicOptions() {
  const topicSelect = document.getElementById('topicSelect');
  if (!topicSelect) return;
  try {
    const grade = state.selectedGrade || state.currentGrade || 'middle';
    const subject = state.selectedSubject || null;
    const { api } = await import('./api.js');
    const data = await api.questions.topics({ grade, subject });
    const apiTopics = Array.isArray(data.topics) ? data.topics : [];
    const fallback = subject ? (STEM_TOPICS_FALLBACK[subject] || []) : [];
    const all = [...apiTopics, ...fallback]
      .map(normalizeTopicSlug)
      .filter(Boolean);
    const topics = [...new Set(all)].sort();

    topicSelect.innerHTML = `<option value="">All topics</option>` + topics
      .map(t => `<option value="${t}">${formatTopicLabel(t)}</option>`)
      .join('');

    if (state.selectedTopic) {
      topicSelect.value = normalizeTopicSlug(state.selectedTopic);
    }
  } catch (err) {
    console.error('Failed to load topics:', err);
    const subject = state.selectedSubject || null;
    const fallback = (subject ? (STEM_TOPICS_FALLBACK[subject] || []) : [])
      .map(normalizeTopicSlug)
      .filter(Boolean);
    const topics = [...new Set(fallback)].sort();
    topicSelect.innerHTML = `<option value="">All topics</option>` + topics
      .map(t => `<option value="${t}">${formatTopicLabel(t)}</option>`)
      .join('');
  }
}

function setupTopicSelector() {
  const topicSelect = document.getElementById('topicSelect');
  if (!topicSelect) return;
  topicSelect.innerHTML = `<option value="">All topics</option>`;
}

// Helper function to ensure grade and subject controls are disabled
function ensureControlsDisabled() {
  if (!isCameraRequiredForLearning()) {
    setLearningControlsEnabled(true);
    setQuestionInteractionLocked(false);
    updateLearningApplyButtonState();
    console.log('🔓 Camera not required. Learning controls remain enabled.');
    return;
  }

  setLearningControlsEnabled(false);
  updateLearningApplyButtonState();
  document.querySelectorAll('.subject-btn').forEach(btn => btn.classList.remove('active'));
  console.log('🔒 Grade, subject, and topic controls locked until real detection is confirmed.');
}

function setupQuestionHandlers() {
  const submitBtn = document.getElementById('submitAnswer');
  const nextBtn = document.getElementById('nextQuestion');
  
  if (submitBtn) {
    submitBtn.addEventListener('click', () => {
      const selectedOption = document.querySelector('input[name="answer"]:checked');
      if (!selectedOption) {
        utils.showNotification('Please select an answer', 'warning');
        return;
      }
      handleQuestionSubmit();
    });
  }
  
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      loadNextQuestion({ fromTimeout: false });
    });
  }

  const applySelectionBtn = document.getElementById('applySelectionBtn');
  if (applySelectionBtn) {
    applySelectionBtn.addEventListener('click', async () => {
      await applyLearningSelection();
    });
  }

  // Topic change handling
  const topicSelect = document.getElementById('topicSelect');
  if (topicSelect) {
    topicSelect.addEventListener('change', (e) => {
      setLearningMode('practice', { preserveAssignedSelection: false });
      updateState({ selectedTopic: normalizeTopicSlug(e.target.value) || null });
      markLearningSelectionPending(state.selectedSubject != null);
    });
  }

  // Questions count selector
  const countSelect = document.getElementById('countSelect');
  if (countSelect) {
    countSelect.addEventListener('change', (e) => {
      setLearningMode('practice', { preserveAssignedSelection: false });
      updateState({ questionsPerSession: parseInt(e.target.value, 10) });
      markLearningSelectionPending(state.selectedSubject != null);
    });
  }

  const difficultySelect = document.getElementById('difficultySelect');
  if (difficultySelect) {
    difficultySelect.addEventListener('change', (e) => {
      setLearningMode('practice', { preserveAssignedSelection: false });
      const selectedPracticeDifficulty = String(e.target.value || 'adaptive').toLowerCase();
      updateState({ selectedPracticeDifficulty });
      markLearningSelectionPending(state.selectedSubject != null);
    });
  }

  const assignedTestSelect = document.getElementById('assignedTestSelect');
  if (assignedTestSelect) {
    assignedTestSelect.addEventListener('change', (e) => {
      const assignmentId = Number(e.target.value || 0);
      selectedAssignedLearning = assignedLearningTests.find(item => Number(item.id) === assignmentId) || null;
      activeAssignedTestSession = null;

      if (selectedAssignedLearning && selectedAssignedLearning.test) {
        setLearningMode('assigned', { preserveAssignedSelection: true });
        updateState({ selectedSubject: selectedAssignedLearning.test.subject?.toLowerCase?.() || state.selectedSubject });
        utils.showNotification('Assigned test selected. Press OK to start assigned learning.', 'info');
      }

      markLearningSelectionPending(Boolean(selectedAssignedLearning));
    });
  }

  const learningModeInputs = document.querySelectorAll('input[name="learningMode"]');
  learningModeInputs.forEach(input => {
    input.addEventListener('change', () => {
      const nextMode = input.value === 'assigned' ? 'assigned' : 'practice';
      if (nextMode === 'practice') {
        setLearningMode('practice', { preserveAssignedSelection: false });
      } else {
        setLearningMode('assigned', { preserveAssignedSelection: true });
        if (!isLearningControlReady()) {
          utils.showNotification('Assigned tests require camera and live face detection. Start camera, then press OK.', 'info');
        }
      }

      markLearningSelectionPending(false);
    });
  });

  const refreshAssignedBtn = document.getElementById('refreshAssignedTests');
  if (refreshAssignedBtn) {
    refreshAssignedBtn.addEventListener('click', async () => {
      await loadAssignedLearningTests();
      utils.showNotification('Assigned tests refreshed.', 'info');
    });
  }

  updateLearningApplyButtonState();
}

function sanitizeQuestionForDisplay(question) {
  if (!question || typeof question !== 'object') return null;

  const text = String(question.text || '').trim();
  if (!text) return null;

  const rawOptions = Array.isArray(question.options) ? question.options : [];
  const options = rawOptions
    .map(option => String(option ?? '').trim())
    .filter(option => option.length > 0);

  if (options.length < 2) {
    return null;
  }

  return {
    ...question,
    text,
    options,
    subject: String(question.subject || state.selectedSubject || 'general').toLowerCase(),
    difficulty: String(question.difficulty || 'medium').toLowerCase(),
  };
}

function normalizeQuestionBatch(questions, maxCount = 50) {
  const rows = Array.isArray(questions) ? questions : [];
  const seen = new Set();
  const cleaned = [];

  for (const row of rows) {
    if (cleaned.length >= maxCount) break;
    const normalized = sanitizeQuestionForDisplay(row);
    if (!normalized) continue;

    const key = normalized.id ? `id:${normalized.id}` : `text:${normalized.text.toLowerCase()}`;
    if (seen.has(key)) continue;

    seen.add(key);
    cleaned.push(normalized);
  }

  return cleaned;
}

function getApiErrorMessage(error, fallback = 'Request failed. Please try again.') {
  const payload = error?.payload || {};
  const serverMessage = payload.error || payload.message || error?.serverMessage || '';
  return String(serverMessage || error?.userMessage || error?.message || fallback);
}

function isAssignmentIntegrityError(error) {
  return String(error?.payload?.code || '').trim().toLowerCase() === 'assignment_integrity_required';
}

function finishPracticeSession() {
  stopQuestionTimer();

  const submitBtn = document.getElementById('submitAnswer');
  const nextBtn = document.getElementById('nextQuestion');
  if (submitBtn) submitBtn.style.display = 'none';
  if (nextBtn) nextBtn.style.display = 'none';

  const feedbackEl = document.getElementById('feedbackMessage');
  if (feedbackEl) {
    feedbackEl.innerHTML = '<div class="alert alert-info"><strong>Session complete.</strong> You reached the selected question limit. Change filters and press OK to start a new set.</div>';
  }

  utils.showNotification('Practice session complete.', 'success');
}

async function loadQuestionsForCurrentSelection() {
  const questionContainer = document.getElementById('questionContainer');
  
  try {
    const { api } = await import('./api.js');

    if (getSelectedLearningMode() === 'assigned') {
      if (!isAssignedLearningMode()) {
        utils.showNotification('Select an assigned test first.', 'warning');
        return;
      }

      const assigned = selectedAssignedLearning;
      const test = assigned.test;

      if (questionContainer) {
        questionContainer.style.display = 'block';
      }

      await api.student.startTest(test.id);
      const testPayload = await api.student.getTestQuestions(test.id);
      const assignedQuestions = Array.isArray(testPayload.questions) ? testPayload.questions : [];

      if (assignedQuestions.length === 0) {
        updateState({ questions: [], currentQuestionIndex: 0 });
        utils.showNotification('Assigned test has no available questions.', 'warning');
        return;
      }

      const mapped = assignedQuestions.map(item => ({
        id: item.id,
        text: item.text,
        options: item.options || [],
        subject: test.subject || state.selectedSubject || 'assigned',
        difficulty: test.difficulty || 'medium',
        __assignedTestId: test.id,
      }));
      const cleanedAssignedQuestions = normalizeQuestionBatch(mapped, assignedQuestions.length || 50);

      if (cleanedAssignedQuestions.length === 0) {
        updateState({ questions: [], currentQuestionIndex: 0 });
        utils.showNotification('Assigned test contains invalid question data (missing options/text).', 'error');
        return;
      }

      activeAssignedTestSession = {
        assignmentId: assigned.id,
        testId: test.id,
      };

      updateState({ questions: cleanedAssignedQuestions, currentQuestionIndex: 0 });
      displayCurrentQuestion();
      utils.showNotification(`Assigned test started: ${test.title}`, 'success');
      return;
    }
    
    // Show question container and loading state
    if (questionContainer) {
      questionContainer.style.display = 'block';
    }
    
    // Initialize adaptive learning for this subject
    adaptiveLearning.initializeSubject(state.selectedSubject);
    if (state.lastAdaptiveSubject !== state.selectedSubject) {
      updateState({
        serverAdaptiveDifficulty: null,
        lastAdaptiveSubject: state.selectedSubject,
      });
    }
    
    // Get recommended difficulty from adaptive learning
    const recommendedDifficulty = adaptiveLearning.getRecommendedDifficulty(state.selectedSubject);
    const serverRecommendedDifficulty = state.serverAdaptiveDifficulty || null;
    
    // Use selectedGrade (from dropdown) instead of currentGrade (user's profile grade)
    const gradeToLoad = state.selectedGrade || state.currentGrade || 'middle';
    
    // Load settings to apply user preferences
    const settings = getEffectiveSettings();
    const configuredDefaultDifficulty = String(settings.defaultDifficulty || 'adaptive').toLowerCase();
    const selectedPracticeDifficulty = getSelectedPracticeDifficulty();
    const selectedOrDefaultDifficulty = selectedPracticeDifficulty !== 'adaptive'
      ? selectedPracticeDifficulty
      : configuredDefaultDifficulty;
    const difficultyToUse = selectedOrDefaultDifficulty === 'adaptive'
      ? (serverRecommendedDifficulty || recommendedDifficulty)
      : selectedOrDefaultDifficulty;
    
    console.log('📚 Loading questions for ' + state.selectedSubject);
    console.log('🎓 User\'s grade:', state.currentGrade);
    console.log('📖 Selected grade:', gradeToLoad);
    console.log('🎚️ Practice difficulty preference:', selectedPracticeDifficulty);
    console.log('📊 Adaptive difficulty (local):', recommendedDifficulty);
    console.log('🤖 Adaptive difficulty (server):', serverRecommendedDifficulty || 'none');
    console.log('⚙️ Using difficulty:', difficultyToUse, '(Configured default:', configuredDefaultDifficulty + ')');
    
    // Respect topic and count selections
    const topic = state.selectedTopic || null;
    const rawCount = state.questionsPerSession || (document.getElementById('countSelect') ? parseInt(document.getElementById('countSelect').value, 10) : (settings.questionsPerSession || 10));
    const count = Math.max(1, Math.min(50, Number.isFinite(rawCount) ? Number(rawCount) : 10));

    const params = {
      grade: gradeToLoad,
      subject: state.selectedSubject,
      difficulty: difficultyToUse,
      limit: count,
      exclude_answered: true,
    };
    
    console.log('🔍 Question API params:', params);

    let response;
    // Fast path: pull from existing generated bank first to avoid request timeout.
    const listParams = { ...params };
    if (topic) listParams.topic = topic;
    response = await api.questions.list(listParams);

    if (response.adaptive && response.adaptive.recommended_difficulty) {
      updateState({ serverAdaptiveDifficulty: response.adaptive.recommended_difficulty });
    }

    console.log('✅ API Response: ' + (response.questions?.length || 0) + ' questions found');
    
    // Fallback 1: Try without difficulty filter if no questions found
    if (!response.questions || response.questions.length === 0) {
      console.log('⚠️ No questions with recommended difficulty, trying all difficulties...');
      const paramsWithoutDifficulty = {
        grade: gradeToLoad,
        subject: state.selectedSubject,
        topic: topic || undefined,
        limit: count,
        exclude_answered: true,
      };
      response = await api.questions.list(paramsWithoutDifficulty);
      console.log('🔄 Retry result: ' + (response.questions?.length || 0) + ' questions found');
    }
    
    // Fallback 2: Try without grade filter if still no questions
    if (!response.questions || response.questions.length === 0) {
      console.log('⚠️ No questions for selected grade, trying all grades...');
      const paramsOnlySubject = {
        subject: state.selectedSubject,
        topic: topic || undefined,
        limit: count,
        exclude_answered: true,
      };
      response = await api.questions.list(paramsOnlySubject);
      console.log('🔄 Retry with all grades: ' + (response.questions?.length || 0) + ' questions found');
    // Final top-up path: if still short and topic selected, trigger generation once.
    if (topic && (!response.questions || response.questions.length < count)) {
      try {
        const generated = await api.questions.generate({
          grade: gradeToLoad,
          subject: state.selectedSubject,
          topic,
          count,
          difficulty: difficultyToUse,
          exclude_answered: true,
        });
        if (generated && Array.isArray(generated.questions) && generated.questions.length > 0) {
          response = generated;
        }
      } catch (genErr) {
        console.warn('Topic generation fallback failed:', genErr);
      }
    }

      
      if (response.questions && response.questions.length > 0) {
        utils.showNotification('No ' + utils.getGradeDisplayName(gradeToLoad) + ' questions available. Showing questions from other grades.', 'info');
      }
    }
    
    const cleanedQuestions = normalizeQuestionBatch(response.questions || [], count);

    if (cleanedQuestions.length > 0) {
      // Store questions in state
      updateState({ questions: cleanedQuestions, currentQuestionIndex: 0 });
      
      // Load first question
      displayCurrentQuestion();
      if (cleanedQuestions.length < count) {
        utils.showNotification(`Loaded ${cleanedQuestions.length} valid questions (requested ${count}).`, 'info');
      } else {
        utils.showNotification('Loaded ' + cleanedQuestions.length + ' ' + state.selectedSubject + ' questions', 'success');
      }
    } else {
      // No questions available at all
      updateState({ questions: [], currentQuestionIndex: 0 });
      
      if (questionContainer) {
        questionContainer.innerHTML = '\n          <div class="empty-state">\n            <div class="empty-state-icon">\n              <i class="fas fa-book-open"></i>\n            </div>\n            <h3 class="empty-state-title">No Questions Available</h3>\n            <p class="empty-state-text">\n              There are currently no ' + state.selectedSubject + ' questions for ' + utils.getGradeDisplayName(gradeToLoad) + ' level.\n              <br>Try selecting a different subject or grade level.\n            </p>\n          </div>\n        ';
      }
      
      utils.showNotification('No ' + state.selectedSubject + ' questions available', 'warning');
    }
  } catch (error) {
    console.error('❌ Failed to load questions:', error);

    if (isAssignmentIntegrityError(error)) {
      const message = getApiErrorMessage(
        error,
        'Assigned tests require active camera and emotion tracking. Please enable camera and keep face detection active.'
      );
      utils.showNotification(message, 'warning');
      return;
    }

    utils.showNotification('Failed to load questions. Please try again.', 'error');
    
    if (questionContainer) {
      questionContainer.innerHTML = '\n        <div class="empty-state">\n          <div class="empty-state-icon">\n            <i class="fas fa-exclamation-triangle"></i>\n          </div>\n          <h3 class="empty-state-title">Error Loading Questions</h3>\n          <p class="empty-state-text">\n            Unable to load questions. Please check your connection and try again.\n          </p>\n        </div>\n      ';
    }
  }
}

function displayCurrentQuestion() {
  if (!state.questions || state.questions.length === 0) {
    console.log('No questions to display');
    return;
  }
  
  const question = state.questions[state.currentQuestionIndex];
  displayQuestion(question);
  ensureTimerIsActiveForCurrentQuestion();
}

function displayQuestion(question) {
  // Update question UI
  const questionTextEl = document.getElementById('questionText');
  const optionsContainer = document.getElementById('answerOptions');
  const categoryEl = document.getElementById('questionCategory');
  const difficultyEl = document.getElementById('difficultyBadge');
  
  if (questionTextEl) questionTextEl.textContent = question.text;
  if (categoryEl) categoryEl.textContent = question.subject;
  if (difficultyEl) {
    difficultyEl.textContent = question.difficulty;
    difficultyEl.className = `difficulty-badge difficulty-${question.difficulty}`;
  }
  
  // Display options
  if (optionsContainer) {
    const options = Array.isArray(question.options) ? question.options : [];
    optionsContainer.innerHTML = options.map((option, index) => `
      <div class="option" data-index="${index}">
        <input type="radio" name="answer" id="option${index}" value="${index}">
        <label for="option${index}">${option}</label>
      </div>
    `).join('');

    if (options.length < 2) {
      optionsContainer.innerHTML = '<div class="alert alert-warning">This question has invalid options and cannot be answered. Please reload your selection.</div>';
    }
    
    // Add click handlers
    const optionNodes = optionsContainer.querySelectorAll('.option');
    optionNodes.forEach(opt => {
      opt.addEventListener('click', () => {
        const radio = opt.querySelector('input[type="radio"]');
        radio.checked = true;
      });
    });
  }
  
  // Show submit button, hide next button
  const submitBtn = document.getElementById('submitAnswer');
  const nextBtn = document.getElementById('nextQuestion');
  if (submitBtn) submitBtn.style.display = 'inline-block';
  if (nextBtn) nextBtn.style.display = 'none';
  
  // Clear feedback
  const feedbackEl = document.getElementById('feedbackMessage');
  if (feedbackEl) feedbackEl.innerHTML = '';
  
  // Track question start time
  updateState({ questionStartTime: Date.now() });
  
  console.log('📝 Displaying question, starting timer...');
  
  // Start question timer if enabled
  startQuestionTimer(`${question.id || 'q'}-${state.currentQuestionIndex}`);
}

function highlightAnswerReview(selectedIndex, correctIndex) {
  const optionNodes = document.querySelectorAll('#answerOptions .option');
  optionNodes.forEach((node, index) => {
    node.classList.remove('option-selected', 'option-correct', 'option-incorrect');
    if (index === selectedIndex) node.classList.add('option-selected');
    if (index === correctIndex) node.classList.add('option-correct');
    if (index === selectedIndex && selectedIndex !== correctIndex) node.classList.add('option-incorrect');
  });
}

function renderAnswerFeedback({ isCorrect, selectedText, correctText, explanationText, timeout }) {
  const feedbackEl = document.getElementById('feedbackMessage');
  if (!feedbackEl) return;

  const statusClass = isCorrect ? 'alert-success' : 'alert-danger';
  const statusTitle = isCorrect ? 'Correct!' : 'Incorrect';
  const timeoutLine = timeout ? '<div class="feedback-line"><strong>Note:</strong> Time expired before selection.</div>' : '';
  const selectedLine = `<div class="feedback-line"><strong>Your answer:</strong> ${selectedText || 'No option selected'}</div>`;
  const correctLine = `<div class="feedback-line"><strong>Correct answer:</strong> ${correctText || 'Not available'}</div>`;
  const explanationLine = explanationText
    ? `<div class="feedback-line"><strong>Explanation:</strong> ${explanationText}</div>`
    : '';

  feedbackEl.innerHTML = `
    <div class="alert ${statusClass} answer-review-card">
      <strong>${statusTitle}</strong>
      ${timeoutLine}
      ${selectedLine}
      ${correctLine}
      ${explanationLine}
    </div>
  `;
}

async function handleQuestionSubmit(options = {}) {
  const { forceTimeoutSubmit = false, autoAdvanceOnTimeout = false } = options;
  const question = state.questions[state.currentQuestionIndex];
  
  if (!question) {
    console.error('No current question');
    return;
  }
  
  const selectedOption = document.querySelector('input[name="answer"]:checked');
  
  if (!selectedOption && !forceTimeoutSubmit) {
    utils.showNotification('Please select an answer', 'warning');
    return;
  }
  
  // Stop the question timer
  stopQuestionTimer();
  
  const selectedIndex = selectedOption ? parseInt(selectedOption.value, 10) : -1;
  const timeSpent = Math.floor((Date.now() - (state.questionStartTime || Date.now())) / 1000);
  const currentEmotion = state.currentEmotion || 'neutral';
  
  try {
    const { api } = await import('./api.js');

    // Submit answer
    const result = isAssignedLearningMode() && activeAssignedTestSession
      ? await api.student.submitTestAnswer(activeAssignedTestSession.testId, {
        question_id: question.id,
        selected_index: selectedIndex,
        time_spent: timeSpent,
      }).then(r => ({
        correct: Boolean(r.is_correct),
        correct_index: Number(r.correct_index ?? -1),
        explanation: '',
        progress: null,
      }))
      : await api.questions.submit(question.id, selectedIndex, timeSpent, currentEmotion);
    
    // Record answer in adaptive learning system with response time and question ID
    const adaptiveResult = adaptiveLearning.recordAnswer(
      question.subject, 
      result.correct, 
      currentEmotion, 
      timeSpent,
      question.id
    );
    
    // Show difficulty change notification if adjusted
    if (adaptiveResult && adaptiveResult.adjusted) {
      const changeType = adaptiveResult.newDifficulty > adaptiveResult.oldDifficulty ? 'increased' : 'decreased';
      const icon = changeType === 'increased' ? '📈' : '📉';
      utils.showNotification(
        `${icon} Difficulty ${changeType} to ${adaptiveResult.newDifficulty}!`,
        changeType === 'increased' ? 'info' : 'warning'
      );
    }

    // Prefer backend adaptive signal (BKT+IRT) when available.
    const backendRecommended = result?.adaptive?.recommended_difficulty || result?.progress?.current_difficulty || null;
    if (backendRecommended) {
      const previousServerDifficulty = state.serverAdaptiveDifficulty;
      updateState({ serverAdaptiveDifficulty: backendRecommended });
      if (previousServerDifficulty && previousServerDifficulty !== backendRecommended) {
        utils.showNotification(`AI adaptation updated: ${previousServerDifficulty} → ${backendRecommended}`, 'info');
      }
    }
    
    // Show streak notification for milestones
    const performance = adaptiveLearning.getSubjectPerformance(question.subject);
    if (performance && result.correct) {
      const streak = performance.currentStreak;
      if (streak === 3) {
        utils.showNotification('🔥 3 in a row! You\'re on fire!', 'success');
      } else if (streak === 5) {
        utils.showNotification('🎯 5 streak! Amazing accuracy!', 'success');
      } else if (streak === 10) {
        utils.showNotification('⭐ 10 STREAK! Absolutely brilliant!', 'success');
      }
    }
    
    // Update subject performance display
    updateSubjectPerformanceWidget();
    
    // Load settings to check if explanations should be shown
    const settings = getEffectiveSettings();
    const explanationText = settings.showExplanations ? (result.explanation || '') : '';
    
    console.log('🎵 Sound settings:', { 
      enabled: settings.enableSoundEffects, 
      correct: result.correct 
    });
    
    // Play sound effects if enabled
    if (settings.enableSoundEffects) {
      console.log('🔊 Attempting to play sound for', result.correct ? 'CORRECT' : 'INCORRECT', 'answer');
      if (result.correct) {
        playSound('correct');
      } else {
        playSound('incorrect');
      }
    } else {
      console.log('🔇 Sound effects disabled in settings');
    }
    
    // Show detailed review feedback and mark options.
    const correctIndex = Number(result.correct_index ?? -1);
    const correctAnswer = correctIndex >= 0 ? question.options[correctIndex] : '';
    const selectedAnswer = selectedIndex >= 0 ? question.options[selectedIndex] : '';
    highlightAnswerReview(selectedIndex, correctIndex);
    if (result.correct) {
      renderAnswerFeedback({
        isCorrect: true,
        selectedText: selectedAnswer,
        correctText: correctAnswer,
        explanationText,
        timeout: forceTimeoutSubmit && !selectedOption
      });
      utils.showNotification('Correct!', 'success');
    } else {
      renderAnswerFeedback({
        isCorrect: false,
        selectedText: selectedAnswer,
        correctText: correctAnswer,
        explanationText,
        timeout: forceTimeoutSubmit && !selectedOption
      });
      utils.showNotification(forceTimeoutSubmit && !selectedOption ? 'Time expired. Marked as incorrect.' : 'Incorrect', 'error');
    }
    
    // Update progress display
    if (result.progress) {
      console.log('Progress updated:', result.progress);
    }
    
    // Hide submit button, show next button
    const submitBtn = document.getElementById('submitAnswer');
    const nextBtn = document.getElementById('nextQuestion');
    if (submitBtn) submitBtn.style.display = 'none';
    if (nextBtn) nextBtn.style.display = autoAdvanceOnTimeout ? 'none' : 'inline-block';
    
    // Disable option selection
    const options = document.querySelectorAll('input[name="answer"]');
    options.forEach(opt => opt.disabled = true);

    if (autoAdvanceOnTimeout) {
      setTimeout(() => {
        loadNextQuestion({ fromTimeout: true });
      }, 350);
    }
    
  } catch (error) {
    console.error('Failed to submit answer:', error);
    if (isAssignmentIntegrityError(error)) {
      utils.showNotification(getApiErrorMessage(error, 'Integrity checks failed. Keep camera and emotion tracking active.'), 'warning');
      return;
    }
    utils.showNotification('Failed to submit answer', 'error');
  }
}

function getDifficultyRank(level) {
  const order = ['easy', 'medium', 'hard'];
  const idx = order.indexOf(String(level || 'medium').toLowerCase());
  return idx === -1 ? 1 : idx;
}

function getAdaptiveNextQuestionIndex() {
  if (!state.questions || state.questions.length === 0) return -1;

  const currentIndex = Number(state.currentQuestionIndex || 0);
  const preferred = String(state.serverAdaptiveDifficulty || '').toLowerCase();
  if (!preferred) return currentIndex + 1;

  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = currentIndex + 1; i < state.questions.length; i++) {
    const row = state.questions[i];
    const distance = Math.abs(getDifficultyRank(row?.difficulty) - getDifficultyRank(preferred));
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
      if (distance === 0) break;
    }
  }

  return bestIndex;
}

function loadNextQuestion(options = {}) {
  const { fromTimeout = false } = options;

  if (!state.questions || state.questions.length === 0) {
    utils.showNotification('No more questions available', 'info');
    return;
  }
  
  const nextIndex = getAdaptiveNextQuestionIndex();
  
  if (Number.isInteger(nextIndex) && nextIndex >= 0 && nextIndex < state.questions.length) {
    updateState({ currentQuestionIndex: nextIndex, questionStartTime: Date.now() });
    displayCurrentQuestion();
    ensureTimerIsActiveForCurrentQuestion();

    if (fromTimeout) {
      utils.showNotification('Next question started. Timer reset.', 'info');
    }
  } else {
    if (isAssignedLearningMode() && activeAssignedTestSession) {
      import('./api.js').then(async ({ api }) => {
        try {
          await api.student.finishTest(activeAssignedTestSession.testId);
          utils.showNotification('Assigned test completed successfully.', 'success');
          activeAssignedTestSession = null;
          selectedAssignedLearning = null;
          const assignedSelect = document.getElementById('assignedTestSelect');
          if (assignedSelect) assignedSelect.value = '';
          await loadAssignedLearningTests();
        } catch (error) {
          console.error('Failed to finish assigned test:', error);
          if (isAssignmentIntegrityError(error)) {
            utils.showNotification(getApiErrorMessage(error, 'Integrity checks failed. Keep camera and emotion tracking active.'), 'warning');
            return;
          }
          utils.showNotification('Failed to finalize assigned test.', 'error');
        }
      });
      return;
    }

    finishPracticeSession();
  }
}

function ensureTimerIsActiveForCurrentQuestion() {
  const settings = getEffectiveSettings();
  if (!settings.enableTimer) {
    stopQuestionTimer();
    return;
  }

  const question = state.questions?.[state.currentQuestionIndex];
  if (!question) return;

  const expectedTimerKey = `${question.id || 'q'}-${state.currentQuestionIndex}`;
  const timerEl = document.getElementById('questionTimer');
  const timerVisible = timerEl && timerEl.style.display !== 'none';

  if (!questionTimerInterval || activeQuestionTimerKey !== expectedTimerKey || !timerVisible) {
    startQuestionTimer(expectedTimerKey);
  }
}

function updateSubjectPerformanceWidget() {
  console.log('📊 updateSubjectPerformanceWidget called');
  
  const performanceGrid = document.getElementById('subjectPerformanceGrid');
  if (!performanceGrid) {
    console.error('❌ Performance grid element (#subjectPerformanceGrid) not found in DOM');
    console.log('Available elements:', document.querySelectorAll('[id*="performance"]'));
    return;
  }
  
  console.log('✅ Performance grid found:', performanceGrid);
  
  const allPerformance = adaptiveLearning.getAllPerformance();
  console.log('📊 Performance data - Subjects:', allPerformance.length, allPerformance);
  
  if (allPerformance.length === 0) {
    performanceGrid.innerHTML = `
      <div style="padding: 30px; text-align: center;">
        <i class="fas fa-chart-line" style="font-size: 3rem; color: var(--primary-color); opacity: 0.3; margin-bottom: 15px;"></i>
        <p style="margin: 0; font-size: 1rem; color: #666; font-weight: 500;">No data in current session</p>
        <small style="display: block; margin-top: 8px; font-size: 0.85rem; color: #999;">
          💡 Start answering questions to see real-time stats<br>
          <span style="font-size: 0.8rem; color: #bbb;">Data resets on page refresh</span>
        </small>
      </div>
    `;
    performanceGrid.style.opacity = '1';
    console.log('✅ Empty state rendered');
    return;
  }
  
  // Add fade-in animation class
  performanceGrid.style.opacity = '0';
  
  performanceGrid.innerHTML = allPerformance.map(perf => {
    const accuracyColor = perf.accuracy >= 80 ? '#28a745' : perf.accuracy >= 60 ? '#ffc107' : '#dc3545';
    const streakIcon = perf.currentStreak >= 3 ? '🔥' : '';
    const isSelected = state.selectedSubject === perf.subject.toLowerCase();
    return `
      <div class="subject-performance-item ${isSelected ? 'selected-subject' : ''}" style="animation: fadeInSlide 0.3s ease-out;">
        <div class="subject-performance-subject">
          ${perf.subject.toUpperCase()} ${streakIcon}
          ${isSelected ? '<span style="color: var(--primary-color); margin-left: 8px;">●</span>' : ''}
        </div>
        <div class="subject-performance-stats">
          <div class="difficulty-badge difficulty-${perf.difficulty}" title="Current difficulty level. Adjusts based on your performance and emotions.">${perf.difficulty.toUpperCase()}</div>
          <div class="performance-accuracy" style="color: ${accuracyColor}; font-weight: 600;" title="Overall accuracy percentage">${perf.accuracy}% accuracy</div>
          <div class="performance-questions" title="Total questions answered">${perf.questionsAnswered} questions</div>
        </div>
        <div class="subject-performance-meta">
          <span class="streak-badge" title="Current streak of correct answers">🎯 ${perf.currentStreak} streak</span>
          <span class="best-streak" title="Best streak achieved">⭐ ${perf.bestStreak} best</span>
          ${perf.avgResponseTime > 0 ? `<span class="response-time" title="Average time per question">⏱️ ${perf.avgResponseTime}s avg</span>` : ''}
        </div>
      </div>
    `;
  }).join('');
  
  // Trigger fade-in animation
  setTimeout(() => {
    performanceGrid.style.opacity = '1';
    performanceGrid.style.transition = 'opacity 0.3s ease-in';
  }, 10);
}

// ===== REPORTS =====
const REPORT_RANGE_DAYS = {
  '7': 7,
  '30': 30,
  '90': 90,
  all: 3650
};

function getSelectedReportRangeValue() {
  const dateFilter = document.getElementById('dateRangeFilter');
  return dateFilter ? dateFilter.value : '30';
}

function getSelectedReportRangeDays() {
  const value = getSelectedReportRangeValue();
  return REPORT_RANGE_DAYS[value] || 30;
}

function getSelectedReportRangeLabel() {
  const value = getSelectedReportRangeValue();
  const labels = {
    '7': 'Last 7 days',
    '30': 'Last 30 days',
    '90': 'Last 90 days',
    all: 'All time'
  };
  return labels[value] || 'Last 30 days';
}

function setExportButtonState(button, isBusy, busyLabel) {
  if (!button) return;
  if (!button.dataset.defaultLabel) {
    button.dataset.defaultLabel = button.innerHTML;
  }
  button.disabled = isBusy;
  button.innerHTML = isBusy ? `<i class="fas fa-spinner fa-spin"></i> ${busyLabel}` : button.dataset.defaultLabel;
}

function getNormalizedReportsExportData() {
  const data = window.reportsData || {};
  const summary = data.summary || {};
  const subjects = Array.isArray(data.subjects) ? data.subjects : [];
  const recommendations = Array.isArray(summary.recommendations) ? summary.recommendations : [];

  const emotionDistribution = Array.isArray(data.emotions?.distribution)
    ? data.emotions.distribution.map(item => ({
      emotion: item.emotion || 'unknown',
      count: Number(item.count || 0),
      percentage: Number(item.percentage || 0),
      avg_confidence: Number(item.avg_confidence || 0)
    }))
    : [];

  const timelineRows = Array.isArray(data.timeline?.daily_stats)
    ? data.timeline.daily_stats
    : [];

  return {
    summary,
    subjects,
    emotionDistribution,
    timelineRows,
    recommendations,
    timelineMeta: data.timeline || {}
  };
}

function normalizeSubjectRows(subjectRows) {
  const rows = Array.isArray(subjectRows) ? subjectRows : [];
  const byKey = new Map();
  const toTitle = (value) => String(value || '')
    .trim()
    .split(/\s+/)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(' ');

  rows.forEach((row) => {
    if (!row) return;
    const raw = String(row.subject || row.name || '').trim();
    if (!raw) return;
    const key = raw.toLowerCase();
    const total = Number(row.total_questions || 0);
    const correct = Number(row.correct_answers || 0);
    const accuracy = Number(row.accuracy || 0);

    if (!byKey.has(key)) {
      byKey.set(key, {
        subject: raw,
        total_questions: total,
        correct_answers: correct,
        accuracy,
        current_difficulty: row.current_difficulty || 'medium',
        difficulty_breakdown: row.difficulty_breakdown || {},
        total_time_spent: Number(row.total_time_spent || 0),
        best_streak: Number(row.best_streak || 0),
        streak: Number(row.streak || 0)
      });
      return;
    }

    const existing = byKey.get(key);
    existing.total_questions += total;
    existing.correct_answers += correct;
    existing.total_time_spent += Number(row.total_time_spent || 0);
    existing.best_streak = Math.max(existing.best_streak, Number(row.best_streak || 0));
    existing.streak = Math.max(existing.streak, Number(row.streak || 0));
    if (row.current_difficulty) existing.current_difficulty = row.current_difficulty;
  });

  return Array.from(byKey.values())
    .map((item) => ({
      ...item,
      subject: toTitle(item.subject),
      accuracy: item.total_questions > 0
        ? Number(((item.correct_answers / item.total_questions) * 100).toFixed(2))
        : Number(item.accuracy || 0)
    }))
    .sort((a, b) => b.total_questions - a.total_questions || b.accuracy - a.accuracy);
}

function normalizeEmotionDistribution(emotionsData) {
  const emotionAliasMap = {
    surprise: 'surprised',
    surprised: 'surprised',
    amazement: 'surprised',
    amazed: 'surprised',
    confusing: 'confused',
    confused: 'confused',
    frustration: 'confused',
    frustrated: 'confused',
    anger: 'angry',
    angry: 'angry',
    joy: 'happy',
    happiness: 'happy',
    happy: 'happy',
    focus: 'focused',
    focused: 'focused',
    engagement: 'focused',
    engaged: 'focused',
    calm: 'neutral',
    neutral: 'neutral',
    bore: 'bored',
    bored: 'bored'
  };

  const list = Array.isArray(emotionsData?.distribution) ? emotionsData.distribution : [];
  const merged = new Map();

  list
    .map((item) => ({
      emotion: String(item.emotion || 'unknown').trim().toLowerCase(),
      count: Number(item.count || 0),
      percentage: Number(item.percentage || 0),
      avg_confidence: Number(item.avg_confidence || 0)
    }))
    .filter(item => item.count > 0)
    .forEach(item => {
      const canonical = emotionAliasMap[item.emotion] || item.emotion;
      if (!merged.has(canonical)) {
        merged.set(canonical, {
          emotion: canonical,
          count: 0,
          avg_confidence_sum: 0
        });
      }
      const bucket = merged.get(canonical);
      bucket.count += item.count;
      bucket.avg_confidence_sum += item.avg_confidence * item.count;
    });

  const total = Array.from(merged.values()).reduce((sum, item) => sum + item.count, 0);

  return Array.from(merged.values())
    .map(item => ({
      emotion: item.emotion,
      count: item.count,
      percentage: total > 0 ? Number(((item.count / total) * 100).toFixed(2)) : 0,
      avg_confidence: item.count > 0 ? Number((item.avg_confidence_sum / item.count).toFixed(2)) : 0
    }))
    .sort((a, b) => b.count - a.count);
}

function normalizeTimelineData(timeline) {
  const base = timeline && typeof timeline === 'object' ? timeline : {};
  const labels = Array.isArray(base.labels) ? base.labels.map(v => String(v ?? '')) : [];
  const correct = Array.isArray(base.correct) ? base.correct.map(v => Number(v || 0)) : [];
  const incorrect = Array.isArray(base.incorrect) ? base.incorrect.map(v => Number(v || 0)) : [];
  const daily_stats = Array.isArray(base.daily_stats) ? base.daily_stats : [];

  // Align chart arrays to avoid truncation/misalignment if backend has partial arrays.
  const rowCount = Math.max(labels.length, correct.length, incorrect.length, daily_stats.length);
  const safeLabels = rowCount > 0
    ? Array.from({ length: rowCount }, (_, i) => labels[i] || daily_stats[i]?.date || `Day ${i + 1}`)
    : [];
  const safeCorrect = rowCount > 0
    ? Array.from({ length: rowCount }, (_, i) => Number(correct[i] ?? daily_stats[i]?.correct ?? 0))
    : [];
  const safeIncorrect = rowCount > 0
    ? Array.from({ length: rowCount }, (_, i) => {
      const direct = incorrect[i];
      if (direct !== undefined && direct !== null) return Number(direct || 0);
      const questions = Number(daily_stats[i]?.questions ?? 0);
      const right = Number(daily_stats[i]?.correct ?? 0);
      return Math.max(0, questions - right);
    })
    : [];

  const rawDifficulty = base.difficulty_breakdown && typeof base.difficulty_breakdown === 'object'
    ? base.difficulty_breakdown
    : {};
  const difficulty_breakdown = {
    easy: Number(rawDifficulty.easy || 0),
    medium: Number(rawDifficulty.medium || 0),
    hard: Number(rawDifficulty.hard || 0),
    unknown: Number(rawDifficulty.unknown || 0)
  };

  return {
    ...base,
    labels: safeLabels,
    correct: safeCorrect,
    incorrect: safeIncorrect,
    daily_stats,
    difficulty_breakdown
  };
}

function setChartLoaderState(loaderId, options = {}) {
  const { state: loaderState = 'hide', message = '' } = options;
  const loader = document.getElementById(loaderId);
  if (!loader) return;

  if (!loader.dataset.defaultContent) {
    loader.dataset.defaultContent = loader.innerHTML;
  }

  if (loaderState === 'loading') {
    loader.innerHTML = loader.dataset.defaultContent;
    loader.classList.remove('hidden');
    loader.style.display = 'flex';
    return;
  }

  if (loaderState === 'empty') {
    loader.classList.remove('hidden');
    loader.style.display = 'flex';
    loader.innerHTML = `<p style="text-align:center;color:#64748b;font-weight:500;max-width:320px;margin:0;">${message || 'No data available for this chart.'}</p>`;
    return;
  }

  loader.classList.add('hidden');
  loader.style.display = 'none';
}

function setChartInsight(insightId, text) {
  const el = document.getElementById(insightId);
  if (!el) return;
  el.innerHTML = text || '<strong>Insight:</strong> Not enough data to derive insights for this chart yet.';
}

function updateReportsInsights(subjects, emotions, timeline) {
  const topSubject = subjects.length > 0 ? subjects[0] : null;
  const weakSubject = subjects.find(s => Number(s.accuracy || 0) < 70);
  const subjectInsight = topSubject
    ? `<strong>Insight:</strong> <strong>${topSubject.subject}</strong> leads with <strong>${Number(topSubject.accuracy || 0).toFixed(1)}%</strong> accuracy across ${topSubject.total_questions} questions.${weakSubject ? ` Focus next on <strong>${weakSubject.subject}</strong> (${Number(weakSubject.accuracy || 0).toFixed(1)}%).` : ''}`
    : '<strong>Insight:</strong> Start solving subject-wise questions to reveal trend comparisons.';
  setChartInsight('subjectPerformanceInsight', subjectInsight);

  const topEmotion = emotions.length > 0 ? emotions[0] : null;
  const emotionInsight = topEmotion
    ? `<strong>Insight:</strong> Dominant state is <strong>${topEmotion.emotion}</strong> (${Number(topEmotion.percentage || 0).toFixed(1)}%). Emotion categories are now strictly distinct and merged from aliases.`
    : '<strong>Insight:</strong> No emotion logs in this period. Turn on camera tracking while learning to populate this analysis.';
  setChartInsight('emotionAnalysisInsight', emotionInsight);

  const difficultyCounts = timeline?.difficulty_breakdown || {};
  const sortedDifficulty = Object.entries(difficultyCounts)
    .filter(([_, count]) => Number(count || 0) > 0)
    .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0));
  const difficultyInsight = sortedDifficulty.length > 0 && Number(sortedDifficulty[0][1] || 0) > 0
    ? `<strong>Insight:</strong> Most attempts are in <strong>${sortedDifficulty[0][0]}</strong> (${sortedDifficulty[0][1]} questions). Balance practice by increasing attempts in underused levels.`
    : '<strong>Insight:</strong> Difficulty distribution will appear after you attempt questions at different levels.';
  setChartInsight('difficultyAnalysisInsight', difficultyInsight);

  const totalCorrect = Array.isArray(timeline?.correct) ? timeline.correct.reduce((a, b) => a + Number(b || 0), 0) : 0;
  const totalIncorrect = Array.isArray(timeline?.incorrect) ? timeline.incorrect.reduce((a, b) => a + Number(b || 0), 0) : 0;
  const total = totalCorrect + totalIncorrect;
  const timelineAcc = total > 0 ? ((totalCorrect / total) * 100).toFixed(1) : '0.0';
  const trend = timeline?.trend || 'insufficient_data';
  const timelineInsight = total > 0
    ? `<strong>Insight:</strong> Timeline accuracy is <strong>${timelineAcc}%</strong> with trend marked as <strong>${trend.replace('_', ' ')}</strong>. Use this with the subject chart to plan focused revision.`
    : '<strong>Insight:</strong> No timeline samples yet. Daily trend appears after answer logs are recorded.';
  setChartInsight('timelineAnalysisInsight', timelineInsight);
}

function buildBusinessRecommendations(summary, subjects, emotions, timeline) {
  const safeSummary = summary || {};
  const safeSubjects = Array.isArray(subjects) ? subjects : [];
  const safeEmotions = Array.isArray(emotions) ? emotions : [];
  const safeTimeline = timeline || {};

  const cards = [];
  const periodDays = Number(safeSummary.period_days || safeTimeline.period_days || 30);
  const activeDays = Number(safeTimeline.active_days || 0);
  const consistencyRate = periodDays > 0 ? (activeDays / periodDays) * 100 : 0;

  const totalEmotionLogs = safeEmotions.reduce((sum, item) => sum + Number(item.count || 0), 0);
  const negativeSet = new Set(['confused', 'bored', 'angry']);
  const negativeCount = safeEmotions
    .filter(item => negativeSet.has(String(item.emotion || '').toLowerCase()))
    .reduce((sum, item) => sum + Number(item.count || 0), 0);
  const negativeRatio = totalEmotionLogs > 0 ? (negativeCount / totalEmotionLogs) * 100 : 0;

  const weakestSubject = [...safeSubjects]
    .filter(s => Number(s.total_questions || 0) >= 3)
    .sort((a, b) => Number(a.accuracy || 0) - Number(b.accuracy || 0))[0];

  const strongestSubject = [...safeSubjects]
    .filter(s => Number(s.total_questions || 0) >= 3)
    .sort((a, b) => Number(b.accuracy || 0) - Number(a.accuracy || 0))[0];

  const recentDaily = Array.isArray(safeTimeline.daily_stats) ? safeTimeline.daily_stats.slice(-7) : [];
  const recentAccuracy = recentDaily
    .filter(d => Number(d.questions || 0) > 0)
    .map(d => Number(d.accuracy || 0));

  const avgRecentAccuracy = recentAccuracy.length > 0
    ? recentAccuracy.reduce((a, b) => a + b, 0) / recentAccuracy.length
    : Number(safeSummary.overall_accuracy || 0);

  if (weakestSubject && Number(weakestSubject.accuracy || 0) < 70) {
    const gap = (70 - Number(weakestSubject.accuracy || 0)).toFixed(1);
    cards.push({
      priority: 1,
      color: 'danger',
      icon: '🎯',
      title: `Close Gap in ${weakestSubject.subject}`,
      description: `Current accuracy is ${Number(weakestSubject.accuracy || 0).toFixed(1)}% across ${weakestSubject.total_questions} questions. Create a focused practice block to recover ${gap}% and reach the 70% benchmark.`,
      action: `Action KPI: Attempt 15 ${weakestSubject.subject} questions in 3 days with >=70% accuracy.`
    });
  }

  if (consistencyRate < 40) {
    cards.push({
      priority: 2,
      color: 'warning',
      icon: '📅',
      title: 'Improve Learning Consistency',
      description: `You were active on ${activeDays}/${periodDays} days (${consistencyRate.toFixed(1)}%). Inconsistent practice usually reduces retention and timeline stability.`,
      action: 'Action KPI: Maintain at least 5 active study days per week for the next 2 weeks.'
    });
  }

  if (negativeRatio >= 45) {
    cards.push({
      priority: 3,
      color: 'warning',
      icon: '🧠',
      title: 'Reduce Cognitive Strain',
      description: `${negativeRatio.toFixed(1)}% of detected emotions are stress-linked (confused/bored/angry). This pattern often correlates with lower answer quality and speed.`,
      action: 'Action KPI: Use 25-minute sessions with 5-minute breaks and track if negative-emotion share drops below 35%.'
    });
  }

  if (strongestSubject && Number(strongestSubject.accuracy || 0) >= 85) {
    cards.push({
      priority: 4,
      color: 'success',
      icon: '🚀',
      title: `Scale Strength in ${strongestSubject.subject}`,
      description: `You are strong in ${strongestSubject.subject} at ${Number(strongestSubject.accuracy || 0).toFixed(1)}%. Convert this into growth by stepping up difficulty.`,
      action: `Action KPI: Move to ${strongestSubject.current_difficulty || 'harder'} level and sustain >=80% over next 10 questions.`
    });
  }

  if (cards.length === 0) {
    cards.push({
      priority: 5,
      color: 'info',
      icon: '📈',
      title: 'Maintain Balanced Performance',
      description: `Current overall accuracy is ${Number(avgRecentAccuracy || 0).toFixed(1)}% with a ${safeTimeline.trend || 'stable'} trend. Performance is balanced without critical risk signals.`,
      action: 'Action KPI: Keep daily practice and target a +5% accuracy gain over the next 14 days.'
    });
  }

  return cards
    .sort((a, b) => a.priority - b.priority)
    .slice(0, 4);
}

async function initReportsPage() {
  if (!ensureStudentRoleSession()) return;
  updateUserInfo();
  setupProfileMenu();
  setupLogout();
  installBackNavigationGuard();
  
  // Setup date range filter
  const dateFilter = document.getElementById('dateRangeFilter');
  if (dateFilter) {
    dateFilter.addEventListener('change', () => {
      loadReportsData(getSelectedReportRangeDays());
    });
  }
  
  // Setup CSV export
  const exportCSVBtn = document.getElementById('exportCSVBtn');
  if (exportCSVBtn) {
    exportCSVBtn.addEventListener('click', async () => {
      setExportButtonState(exportCSVBtn, true, 'Exporting...');
      try {
        await exportReportsToCSV();
      } finally {
        setExportButtonState(exportCSVBtn, false, 'Export CSV');
      }
    });
  }
  
  // Setup PDF export
  const exportPDFBtn = document.getElementById('exportPDFBtn');
  if (exportPDFBtn) {
    exportPDFBtn.addEventListener('click', async () => {
      setExportButtonState(exportPDFBtn, true, 'Generating...');
      try {
        await exportReportsToPDF();
      } finally {
        setExportButtonState(exportPDFBtn, false, 'Export PDF');
      }
    });
  }
  
  // Load initial reports data
  await loadReportsData(getSelectedReportRangeDays());
}

async function loadReportsData(days = 30) {
  const { chartRenderer } = await import('./chart-renderer.js');
  
  // Show skeleton loaders
  ['chartLoader1', 'chartLoader2', 'chartLoader3', 'chartLoader4'].forEach(id => {
    setChartLoaderState(id, { state: 'loading' });
  });
  
  try {
    const { api } = await import('./api.js');
    
    // Load all reports data
    const [summary, subjectsData, emotionsData, timeline, integrity] = await Promise.all([
      api.reports.getSummary(days),
      api.reports.getSubjects(days),
      api.reports.getEmotions(days),
      api.reports.getTimeline(days),
      api.reports.getIntegrity(days).catch(() => null)
    ]);

    if (!summary || typeof summary !== 'object') {
      throw new Error('Invalid summary response from reports API');
    }

    const normalizedSubjects = normalizeSubjectRows(subjectsData?.subjects);
    const normalizedEmotionDistribution = normalizeEmotionDistribution(emotionsData);
    let normalizedTimeline = normalizeTimelineData(timeline);

    if (integrity && integrity.timeline) {
      const hasIntegrityMismatch = Array.isArray(integrity.mismatches) && integrity.mismatches.length > 0;
      const backendTimeline = normalizeTimelineData(integrity.timeline);

      // Always trust canonical integrity payload for timeline/difficulty rendering.
      normalizedTimeline = {
        ...normalizedTimeline,
        ...backendTimeline
      };

      console.info('Reports integrity check', {
        ok: Boolean(integrity.ok),
        mismatches: integrity.mismatches || [],
        totals: integrity.totals || {}
      });

      if (hasIntegrityMismatch) {
        utils.showNotification('Reports data mismatch detected and auto-corrected using canonical integrity data.', 'warning');
      }
    }
    const recommendationCards = buildBusinessRecommendations(
      summary,
      normalizedSubjects,
      normalizedEmotionDistribution,
      normalizedTimeline
    );

    console.info('Reports API connected and normalized', {
      rangeDays: days,
      subjects: normalizedSubjects.length,
      emotions: normalizedEmotionDistribution.length,
      timelinePoints: normalizedTimeline.labels.length
    });
    
    console.log('📥 Data received:', { summary, subjectsData, emotionsData, timeline });
    
    // Check if user has any data
    const hasData = summary && summary.total_questions > 0;
    const emptyState = document.getElementById('reportsEmptyState');
    
    if (!hasData) {
      if (emptyState) emptyState.style.display = 'flex';
      document.querySelectorAll('.report-section').forEach(section => {
        section.style.display = 'none';
      });
      // Hide loaders
      ['chartLoader1', 'chartLoader2', 'chartLoader3', 'chartLoader4'].forEach(id => {
        const loader = document.getElementById(id);
        if (loader) {
          loader.classList.add('hidden');
          loader.style.display = 'none';
        }
      });
      return;
    }
    
    if (emptyState) emptyState.style.display = 'none';
    document.querySelectorAll('.report-section').forEach(section => {
      section.style.display = 'block';
    });
    
    // Update summary box
    const summaryContent = document.getElementById('summaryContent');
    if (summaryContent && summary) {
      const topSummaryRecommendations = recommendationCards.slice(0, 2);
      summaryContent.innerHTML = `
        <div class=\"row\">
          <div class=\"col-md-6\">
            <p><strong>📊 Overall Accuracy:</strong> ${summary.overall_accuracy.toFixed(1)}%</p>
            <p><strong>❓ Total Questions:</strong> ${summary.total_questions}</p>
            <p><strong>📚 Most Practiced:</strong> ${summary.most_practiced_subject || 'N/A'}</p>
          </div>
          <div class=\"col-md-6\">
            <p><strong>⏱️ Time Spent:</strong> ${summary.total_time_minutes} minutes</p>
            <p><strong>📈 Improvement:</strong> ${summary.improvement > 0 ? '+' : ''}${summary.improvement.toFixed(1)}%</p>
            <p><strong>🎯 Current Streak:</strong> ${summary.current_streak || 0}</p>
          </div>
        </div>
        ${topSummaryRecommendations.length > 0 ? `
          <div class=\"mt-3\">
            <strong>💡 Recommendations:</strong>
            <ul>${topSummaryRecommendations.map(rec => `<li><strong>${rec.title}:</strong> ${rec.action}</li>`).join('')}</ul>
          </div>
        ` : ''}
      `;
    }
    
    // Render charts with logging
    console.log('🎨 Rendering charts...');
    
    if (normalizedSubjects.length > 0) {
      console.log('📈 Rendering subject performance chart with:', normalizedSubjects);
      chartRenderer.renderSubjectPerformance('subjectPerformanceCanvas', normalizedSubjects);
      setChartLoaderState('chartLoader1', { state: 'hide' });
    } else {
      setChartLoaderState('chartLoader1', { state: 'empty', message: 'No subject-wise performance data yet. Answer more questions to unlock this chart.' });
    }
    
    if (normalizedEmotionDistribution.length > 0) {
      console.log('😊 Rendering emotion distribution chart with:', normalizedEmotionDistribution);
      chartRenderer.renderEmotionDistribution('emotionDistributionCanvas', normalizedEmotionDistribution);
      setChartLoaderState('chartLoader2', { state: 'hide' });
    } else {
      setChartLoaderState('chartLoader2', { state: 'empty', message: 'No emotion logs found in this period.' });
    }
    
    if (normalizedTimeline && normalizedTimeline.difficulty_breakdown) {
      console.log('🎯 Rendering difficulty chart with:', normalizedTimeline.difficulty_breakdown);
      chartRenderer.renderDifficultyProgression('difficultyAnalysisCanvas', normalizedTimeline.difficulty_breakdown);
      setChartLoaderState('chartLoader3', { state: 'hide' });
    } else {
      setChartLoaderState('chartLoader3', { state: 'empty', message: 'Difficulty distribution is not available for this period.' });
    }
    
    // Render timeline chart
    console.log('📊 Timeline data received:', normalizedTimeline);
    
    if (normalizedTimeline && normalizedTimeline.labels && normalizedTimeline.labels.length > 0) {
      console.log('📅 Rendering timeline chart with:', {
        labels: normalizedTimeline.labels,
        correct: normalizedTimeline.correct,
        incorrect: normalizedTimeline.incorrect,
        totalDays: normalizedTimeline.labels.length
      });
      
      const chartResult = chartRenderer.renderQuestionsTimeline('learningTimelineCanvas', normalizedTimeline);
      console.log('Chart render result:', chartResult ? 'Success' : 'Failed');
      setChartLoaderState('chartLoader4', { state: 'hide' });
    } else {
      console.warn('⚠️ Timeline data missing or empty:', {
        hasTimeline: !!normalizedTimeline,
        hasLabels: !!(normalizedTimeline && normalizedTimeline.labels),
        labelCount: normalizedTimeline?.labels?.length || 0
      });
      setChartLoaderState('chartLoader4', { state: 'empty', message: 'No timeline data available yet. Complete more questions to see your progress over time.' });
    }

    updateReportsInsights(normalizedSubjects, normalizedEmotionDistribution, normalizedTimeline);
    
    // Populate personalized recommendations using weighted business rules
    const improvementCards = document.getElementById('improvementCards');
    if (improvementCards) {
      improvementCards.innerHTML = recommendationCards.map(rec => `
        <div class="improvement-card improvement-card-${rec.color}">
          <div class="improvement-icon">${rec.icon}</div>
          <h4 class="improvement-title">${rec.title}</h4>
          <p class="improvement-description">${rec.description}</p>
          <div class="improvement-action">${rec.action}</div>
        </div>
      `).join('');
    }
    
    // Store data for CSV export
    window.reportsData = {
      summary,
      subjects: normalizedSubjects,
      emotions: { ...(emotionsData || {}), distribution: normalizedEmotionDistribution },
      timeline: normalizedTimeline
    };
    
    console.log('📊 Reports loaded successfully');
  } catch (error) {
    console.error('Failed to load reports:', error);
    utils.showNotification('Failed to load reports data', 'error');
    
    // Hide loaders on error
    ['chartLoader1', 'chartLoader2', 'chartLoader3', 'chartLoader4'].forEach(id => setChartLoaderState(id, { state: 'hide' }));
  }
}

async function exportReportsToCSV() {
  if (!window.reportsData) {
    utils.showNotification('No data to export', 'warning');
    return;
  }

  const {
    summary,
    subjects,
    emotionDistribution,
    timelineRows,
    recommendations,
    timelineMeta
  } = getNormalizedReportsExportData();
  
  // Helper to escape CSV values
  const escapeCSV = (value) => {
    if (value === null || value === undefined) return '';
    const str = String(value);
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };
  
  const rangeLabel = getSelectedReportRangeLabel();

  // Create CSV content with robust data formatting
  let csv = '';
  
  // Header
  csv += 'Elevate Learning Platform - Progress Report\n';
  csv += `Generated,${escapeCSV(new Date().toLocaleString())}\n`;
  csv += `User,${escapeCSV(state.user?.name || 'Student')}\n`;
  csv += `Range,${escapeCSV(rangeLabel)}\n`;
  csv += '\n';
  
  // Summary section
  csv += 'LEARNING SUMMARY\n';
  csv += 'Metric,Value\n';
  csv += `Total Questions Attempted,${summary.total_questions || 0}\n`;
  csv += `Correct Answers,${summary.correct_answers || 0}\n`;
  csv += `Overall Accuracy,${(summary.overall_accuracy || 0).toFixed(1)}%\n`;
  csv += `Time Spent (minutes),${summary.total_time_minutes || 0}\n`;
  csv += `Improvement Rate,${(summary.improvement || 0).toFixed(1)}%\n`;
  csv += `Current Streak,${summary.current_streak || 0}\n`;
  csv += `Active Days,${timelineMeta.active_days || 0}\n`;
  csv += `Timeline Trend,${escapeCSV(timelineMeta.trend || 'insufficient_data')}\n`;
  csv += '\n';
  
  // Subject performance section
  csv += 'SUBJECT PERFORMANCE\n';
  csv += 'Subject,Total Questions,Correct Answers,Accuracy (%),Current Difficulty,Average Time (s)\n';
  if (subjects && subjects.length > 0) {
    subjects.forEach(s => {
      csv += `${escapeCSV(s.subject)},`;
      csv += `${s.total_questions || 0},`;
      csv += `${s.correct_answers || 0},`;
      csv += `${(s.accuracy || 0).toFixed(1)},`;
      csv += `${escapeCSV(s.current_difficulty || 'N/A')},`;
      csv += `${s.total_time_spent ? (Number(s.total_time_spent) / Math.max(1, Number(s.total_questions || 1))).toFixed(1) : 0}\n`;
    });
  } else {
    csv += 'No subject data available\n';
  }
  csv += '\n';
  
  // Emotion analysis section
  if (emotionDistribution.length > 0) {
    csv += 'EMOTION ANALYSIS\n';
    csv += 'Emotion,Count,Percentage (%),Average Confidence\n';
    emotionDistribution.forEach(item => {
      csv += `${escapeCSV(item.emotion)},${item.count},${item.percentage.toFixed(1)},${item.avg_confidence.toFixed(2)}\n`;
    });
    csv += '\n';
  }
  
  // Timeline section
  if (timelineRows.length > 0) {
    csv += 'DAILY PROGRESS TIMELINE\n';
    csv += 'Date,Questions Attempted,Correct,Accuracy (%),Time Spent (minutes)\n';
    timelineRows.forEach(entry => {
      csv += `${escapeCSV(entry.date)},`;
      csv += `${entry.questions || 0},`;
      csv += `${entry.correct || 0},`;
      csv += `${(entry.accuracy || 0).toFixed(1)},`;
      csv += `${(entry.time_minutes || 0).toFixed(1)}\n`;
    });
    csv += '\n';
  }

  if (recommendations.length > 0) {
    csv += 'RECOMMENDATIONS\n';
    recommendations.forEach((item, index) => {
      csv += `${index + 1},${escapeCSV(item)}\n`;
    });
    csv += '\n';
  }
  
  // Footer
  csv += '\n';
  csv += 'Report generated by Elevate Learning Platform\n';
  csv += 'For more details, visit your dashboard\n';
  
  // Download CSV
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `elevate-report-${new Date().toISOString().split('T')[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
  
  utils.showNotification('Report exported to CSV successfully!', 'success');
}

async function exportReportsToPDF() {
  if (!window.reportsData) {
    utils.showNotification('No data to export', 'warning');
    return;
  }

  const {
    summary,
    subjects,
    emotionDistribution,
    timelineRows,
    recommendations,
    timelineMeta
  } = getNormalizedReportsExportData();
  
  // Check if jsPDF is loaded
  if (typeof window.jspdf === 'undefined') {
    utils.showNotification('PDF library not loaded. Please refresh the page.', 'error');
    return;
  }
  
  try {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 44;
    const contentWidth = pageWidth - margin * 2;
    const lineHeight = 15;
    let yPos = margin;

    const ensureSpace = (neededHeight = 24) => {
      if (yPos + neededHeight > pageHeight - margin) {
        doc.addPage();
        yPos = margin;
      }
    };

    const sectionTitle = (title) => {
      ensureSpace(30);
      doc.setDrawColor(37, 99, 235);
      doc.setLineWidth(2);
      doc.line(margin, yPos, margin + 20, yPos);
      yPos += 14;
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(13);
      doc.setTextColor(17, 24, 39);
      doc.text(title, margin, yPos);
      yPos += 12;
    };

    const drawKeyValues = (pairs) => {
      const colA = margin;
      const colB = margin + contentWidth / 2 + 8;
      const rows = Math.ceil(pairs.length / 2);
      ensureSpace(rows * lineHeight + 12);
      doc.setFontSize(10);
      for (let i = 0; i < rows; i++) {
        const left = pairs[i * 2];
        const right = pairs[i * 2 + 1];
        if (left) {
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(55, 65, 81);
          doc.text(`${left[0]}:`, colA, yPos);
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(17, 24, 39);
          doc.text(String(left[1]), colA + 126, yPos);
        }
        if (right) {
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(55, 65, 81);
          doc.text(`${right[0]}:`, colB, yPos);
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(17, 24, 39);
          doc.text(String(right[1]), colB + 92, yPos);
        }
        yPos += lineHeight;
      }
      yPos += 8;
    };

    const drawSimpleTable = (headers, rows, columnWidths) => {
      const headerHeight = 22;
      const rowHeight = 18;
      ensureSpace(headerHeight + rowHeight);

      doc.setFillColor(239, 246, 255);
      doc.rect(margin, yPos, contentWidth, headerHeight, 'F');
      doc.setDrawColor(209, 213, 219);
      doc.rect(margin, yPos, contentWidth, headerHeight);

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9);
      doc.setTextColor(30, 41, 59);
      let x = margin + 6;
      headers.forEach((header, index) => {
        doc.text(header, x, yPos + 14);
        x += columnWidths[index];
      });
      yPos += headerHeight;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);
      rows.forEach((row, rowIndex) => {
        ensureSpace(rowHeight + 6);
        if (rowIndex % 2 === 0) {
          doc.setFillColor(249, 250, 251);
          doc.rect(margin, yPos, contentWidth, rowHeight, 'F');
        }
        doc.setDrawColor(229, 231, 235);
        doc.rect(margin, yPos, contentWidth, rowHeight);
        let colX = margin + 6;
        row.forEach((cell, index) => {
          const text = String(cell ?? '');
          const maxWidth = Math.max(10, columnWidths[index] - 10);
          const clipped = doc.splitTextToSize(text, maxWidth)[0] || '';
          doc.text(clipped, colX, yPos + 12);
          colX += columnWidths[index];
        });
        yPos += rowHeight;
      });

      yPos += 10;
    };

    const drawRecommendations = (items) => {
      if (!items.length) return;
      sectionTitle('Recommendations');
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(10);
      doc.setTextColor(31, 41, 55);
      items.forEach((item, index) => {
        const wrapped = doc.splitTextToSize(`${index + 1}. ${item}`, contentWidth - 14);
        ensureSpace(wrapped.length * lineHeight + 4);
        doc.text(wrapped, margin + 8, yPos);
        yPos += wrapped.length * lineHeight;
      });
      yPos += 6;
    };

    const addChartImage = (canvasId, title) => {
      const canvas = document.getElementById(canvasId);
      if (!canvas || canvas.width === 0 || canvas.height === 0) return;
      ensureSpace(220);
      sectionTitle(title);
      const imgWidth = contentWidth;
      const imgHeight = Math.min(220, imgWidth * (canvas.height / canvas.width));
      ensureSpace(imgHeight + 8);
      const imgData = canvas.toDataURL('image/png', 1.0);
      doc.addImage(imgData, 'PNG', margin, yPos, imgWidth, imgHeight, undefined, 'FAST');
      yPos += imgHeight + 12;
    };

    const now = new Date();
    const rangeLabel = getSelectedReportRangeLabel();
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(20);
    doc.setTextColor(17, 24, 39);
    doc.text('Elevate Learning Report', margin, yPos);
    yPos += 20;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(75, 85, 99);
    doc.text(`Generated: ${now.toLocaleString()}`, margin, yPos);
    yPos += 14;
    doc.text(`Student: ${state.user?.name || 'Student'}`, margin, yPos);
    yPos += 14;
    doc.text(`Range: ${rangeLabel}`, margin, yPos);
    yPos += 16;

    sectionTitle('Summary');
    drawKeyValues([
      ['Total Questions', Number(summary.total_questions || 0)],
      ['Correct Answers', Number(summary.correct_answers || 0)],
      ['Overall Accuracy', `${Number(summary.overall_accuracy || 0).toFixed(1)}%`],
      ['Time Spent', `${Number(summary.total_time_minutes || 0).toFixed(1)} min`],
      ['Improvement', `${Number(summary.improvement || 0).toFixed(1)}%`],
      ['Current Streak', Number(summary.current_streak || 0)],
      ['Active Days', Number(timelineMeta.active_days || 0)],
      ['Trend', timelineMeta.trend || 'insufficient_data']
    ]);

    if (subjects.length > 0) {
      sectionTitle('Subject Performance');
      const rows = subjects.map((s) => [
        s.subject || 'N/A',
        Number(s.total_questions || 0),
        Number(s.correct_answers || 0),
        `${Number(s.accuracy || 0).toFixed(1)}%`,
        s.current_difficulty || 'N/A'
      ]);
      drawSimpleTable(
        ['Subject', 'Questions', 'Correct', 'Accuracy', 'Difficulty'],
        rows,
        [140, 80, 70, 78, 90]
      );
    }

    if (emotionDistribution.length > 0) {
      sectionTitle('Emotion Analysis');
      const rows = emotionDistribution.map((e) => [
        e.emotion,
        e.count,
        `${e.percentage.toFixed(1)}%`,
        e.avg_confidence.toFixed(2)
      ]);
      drawSimpleTable(
        ['Emotion', 'Count', 'Percent', 'Avg Confidence'],
        rows,
        [170, 95, 95, 120]
      );
    }

    if (timelineRows.length > 0) {
      sectionTitle('Daily Timeline');
      const rows = timelineRows.map((row) => [
        row.date,
        Number(row.questions || 0),
        Number(row.correct || 0),
        `${Number(row.accuracy || 0).toFixed(1)}%`,
        `${Number(row.time_minutes || 0).toFixed(1)}m`
      ]);
      drawSimpleTable(
        ['Date', 'Questions', 'Correct', 'Accuracy', 'Time'],
        rows,
        [120, 76, 70, 84, 76]
      );
    }

    drawRecommendations(recommendations);

    addChartImage('subjectPerformanceCanvas', 'Subject Performance Chart');
    addChartImage('emotionDistributionCanvas', 'Emotion Distribution Chart');
    addChartImage('difficultyAnalysisCanvas', 'Difficulty Distribution Chart');
    addChartImage('learningTimelineCanvas', 'Learning Timeline Chart');

    const totalPages = doc.getNumberOfPages();
    for (let page = 1; page <= totalPages; page++) {
      doc.setPage(page);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(107, 114, 128);
      doc.text(`Generated by Elevate | Page ${page} of ${totalPages}`, pageWidth / 2, pageHeight - 18, { align: 'center' });
    }

    doc.save(`elevate-report-${new Date().toISOString().split('T')[0]}.pdf`);
    utils.showNotification('Report exported to PDF successfully!', 'success');
    
  } catch (error) {
    console.error('PDF export error:', error);
    utils.showNotification('Failed to generate PDF. Please try again.', 'error');
  }
}

// ===== PROFILE =====
function initProfilePage() {
  if (!auth.requireAuth()) return;
  updateUserInfo();
  setupProfileMenu();
  setupLogout();
  installBackNavigationGuard();
}

// ===== SETTINGS =====
function initSettingsPage() {
  if (!auth.requireAuth()) return;
  updateUserInfo();
  setupProfileMenu();
  setupLogout();
  installBackNavigationGuard();
  
  loadSettings().catch((error) => {
    console.warn('Failed to load settings from server, using local cache:', error);
  });
  loadProfileInfo();
  setupSettingsHandlers();
}

async function loadProfileInfo() {
  try {
    const { api } = await import('./api.js');
    const profileData = await api.auth.getProfile();
    
    if (profileData && profileData.user) {
      const user = profileData.user;
      
      document.getElementById('profileName').textContent = user.name || 'N/A';
      document.getElementById('profileEmail').textContent = user.email || 'N/A';

      document.getElementById('profileGrade').textContent = utils.getGradeDisplayName(user.grade) || 'N/A';
      
      const joinedDate = user.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A';
      document.getElementById('profileJoined').textContent = joinedDate;
    }
  } catch (error) {
    console.error('Failed to load profile:', error);
  }
}

// Default settings
const DEFAULT_SETTINGS = {
  // Question Timer
  enableTimer: false,
  timerDuration: 60,
  autoSubmit: false,
  
  // Learning Preferences
  defaultDifficulty: 'adaptive',
  questionsPerSession: 10,
  showExplanations: true,
  
  // Camera & Emotion Detection
  requireCamera: true,
  enableEmotionFeedback: true,
  detectionFrequency: 'medium',
  
  // Notifications
  enableNotifications: true,
  enableSoundEffects: false
};

function getEffectiveSettings() {
  return state.userSettings || storage.load('userSettings') || { ...DEFAULT_SETTINGS };
}

async function syncUserSettingsFromServer() {
  try {
    const { api } = await import('./api.js');
    const response = await api.settings.get();
    const localSettings = storage.load('userSettings') || state.userSettings || {};
    // Local state wins for immediate UX consistency when user just changed settings.
    const merged = { ...DEFAULT_SETTINGS, ...(response.settings || {}), ...localSettings };
    storage.save('userSettings', merged);
    updateState({ userSettings: merged });
    return merged;
  } catch (error) {
    const fallback = getEffectiveSettings();
    updateState({ userSettings: fallback });
    return fallback;
  }
}

async function loadSettings() {
  const savedSettings = await syncUserSettingsFromServer();
  
  // Merge with defaults in case new settings were added
  const settings = { ...DEFAULT_SETTINGS, ...savedSettings };
  
  // Apply settings to UI
  document.getElementById('enableTimerCheckbox').checked = settings.enableTimer;
  document.getElementById('timerDuration').value = settings.timerDuration;
  document.getElementById('timerDurationDisplay').textContent = settings.timerDuration + 's';
  document.getElementById('autoSubmitCheckbox').checked = settings.autoSubmit;
  
  document.getElementById('defaultDifficulty').value = settings.defaultDifficulty;
  document.getElementById('questionsPerSession').value = settings.questionsPerSession;
  document.getElementById('showExplanations').checked = settings.showExplanations;
  
  document.getElementById('requireCamera').checked = settings.requireCamera;
  document.getElementById('enableEmotionFeedback').checked = settings.enableEmotionFeedback;
  document.getElementById('detectionFrequency').value = settings.detectionFrequency;
  
  document.getElementById('enableNotifications').checked = settings.enableNotifications;
  document.getElementById('enableSoundEffects').checked = settings.enableSoundEffects;
  
  // Update timer duration setting visibility
  updateTimerSettingVisibility(settings.enableTimer);
  updateEmotionSettingsDependency(settings.requireCamera, settings.enableEmotionFeedback);
  
  // Store in state for access from other pages
  updateState({ userSettings: settings });
  
  console.log('Settings loaded:', settings);
}

function setupSettingsHandlers() {
  // Helper function to save settings in real-time
  const saveSettingRealTime = async (settingKey, value) => {
    const currentSettings = getEffectiveSettings();
    currentSettings[settingKey] = value;
    storage.save('userSettings', currentSettings);
    updateState({ userSettings: currentSettings });

    try {
      const { api } = await import('./api.js');
      const response = await api.settings.update({ [settingKey]: value });
      const merged = { ...DEFAULT_SETTINGS, ...(response.settings || currentSettings) };
      storage.save('userSettings', merged);
      updateState({ userSettings: merged });
    } catch (error) {
      console.warn(`Failed to persist setting ${settingKey} to backend:`, error);
    }
    
    // Show subtle success feedback
    const elem = document.activeElement;
    if (elem) {
      const originalBorder = elem.style.border;
      elem.style.border = '2px solid #10b981';
      setTimeout(() => {
        elem.style.border = originalBorder;
      }, 500);
    }
  };
  
  // Timer duration slider - real-time update
  const timerDuration = document.getElementById('timerDuration');
  const timerDurationDisplay = document.getElementById('timerDurationDisplay');
  
  timerDuration.addEventListener('input', (e) => {
    timerDurationDisplay.textContent = e.target.value + 's';
  });
  
  timerDuration.addEventListener('change', async (e) => {
    await saveSettingRealTime('timerDuration', parseInt(e.target.value, 10));
  });
  
  // Enable timer checkbox - real-time
  const enableTimerCheckbox = document.getElementById('enableTimerCheckbox');
  enableTimerCheckbox.addEventListener('change', async (e) => {
    updateTimerSettingVisibility(e.target.checked);
    await saveSettingRealTime('enableTimer', e.target.checked);
  });
  
  // Auto-submit checkbox - real-time
  const autoSubmitCheckbox = document.getElementById('autoSubmitCheckbox');
  autoSubmitCheckbox.addEventListener('change', async (e) => {
    await saveSettingRealTime('autoSubmit', e.target.checked);
  });
  
  // Default difficulty - real-time
  const defaultDifficulty = document.getElementById('defaultDifficulty');
  defaultDifficulty.addEventListener('change', async (e) => {
    await saveSettingRealTime('defaultDifficulty', e.target.value);
  });
  
  // Questions per session - real-time
  const questionsPerSession = document.getElementById('questionsPerSession');
  questionsPerSession.addEventListener('change', async (e) => {
    const value = parseInt(e.target.value);
    if (value >= 5 && value <= 50) {
      await saveSettingRealTime('questionsPerSession', value);
    } else {
      utils.showNotification('Please enter a value between 5 and 50', 'warning');
      e.target.value = 10;
    }
  });
  
  // Show explanations - real-time
  const showExplanations = document.getElementById('showExplanations');
  showExplanations.addEventListener('change', async (e) => {
    await saveSettingRealTime('showExplanations', e.target.checked);
  });
  
  // Require camera - real-time
  const requireCamera = document.getElementById('requireCamera');
  requireCamera.addEventListener('change', async (e) => {
    await saveSettingRealTime('requireCamera', e.target.checked);
    updateEmotionSettingsDependency(e.target.checked, getEffectiveSettings().enableEmotionFeedback);
    
    // Apply the change immediately - show message based on current page
    if (e.target.checked) {
      utils.showNotification('Practice mode now requires camera. Assigned tests always require camera and emotion tracking.', 'info');
    } else {
      utils.showNotification('Practice mode can run without camera. Assigned tests still require camera and emotion tracking.', 'info');
      if (currentShellRoute === 'learning' || window.location.pathname.includes('learning.html')) {
        updateLearningStatusCard();
        updateLearningApplyButtonState();
      }
    }
  });
  
  // Enable emotion feedback - real-time
  const enableEmotionFeedback = document.getElementById('enableEmotionFeedback');
  enableEmotionFeedback.addEventListener('change', async (e) => {
    await saveSettingRealTime('enableEmotionFeedback', e.target.checked);
    updateEmotionSettingsDependency(getEffectiveSettings().requireCamera, e.target.checked);

    if (!e.target.checked) {
      emotionDetector.resetFeedbackState();
      utils.showNotification('Emotion-based feedback disabled.', 'info');
    } else {
      utils.showNotification('Emotion-based feedback enabled.', 'success');
    }

    // Apply feedback-mode changes immediately to live detection cadence.
    if (state.cameraActive && (state.emotionDetectionInterval || emotionDetector.animationFrameId)) {
      emotionDetector.stopDetection();
      setTimeout(() => emotionDetector.startDetection(), 200);
    }
  });
  
  // Detection frequency - real-time
  const detectionFrequency = document.getElementById('detectionFrequency');
  detectionFrequency.addEventListener('change', async (e) => {
    if (!getEffectiveSettings().enableEmotionFeedback) {
      utils.showNotification('Enable emotion-based feedback to change detection frequency.', 'warning');
      return;
    }

    await saveSettingRealTime('detectionFrequency', e.target.value);
    // If camera is active, restart detection with new frequency
    if (state.cameraActive && (state.emotionDetectionInterval || emotionDetector.animationFrameId)) {
      utils.showNotification(`Detection frequency updated to ${e.target.value}`, 'info');
      emotionDetector.stopDetection();
      setTimeout(() => emotionDetector.startDetection(), 500);
    }
  });
  
  // Enable notifications - real-time
  const enableNotifications = document.getElementById('enableNotifications');
  enableNotifications.addEventListener('change', async (e) => {
    if (!e.target.checked) {
      // Show confirmation before notifications are disabled.
      utils.showNotification('Notifications disabled.', 'info');
    }

    await saveSettingRealTime('enableNotifications', e.target.checked);

    if (e.target.checked) {
      utils.showNotification('Notifications enabled.', 'success');
    }
  });
  
  // Enable sound effects - real-time
  const enableSoundEffects = document.getElementById('enableSoundEffects');
  enableSoundEffects.addEventListener('change', async (e) => {
    await saveSettingRealTime('enableSoundEffects', e.target.checked);

    if (e.target.checked) {
      utils.showNotification('Sound effects enabled.', 'success');
      playSound('correct');
    } else {
      utils.showNotification('Sound effects disabled.', 'info');
    }
  });
  
  // Test sound button
  const testSoundBtn = document.getElementById('testSoundBtn');
  if (testSoundBtn) {
    testSoundBtn.addEventListener('click', () => {
      console.log('🎵 Test sound button clicked');
      playSound('correct');
      setTimeout(() => playSound('incorrect'), 400);
    });
  }
  
  // Reset settings button
  const resetBtn = document.getElementById('resetSettingsBtn');
  resetBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      storage.save('userSettings', DEFAULT_SETTINGS);
      updateState({ userSettings: { ...DEFAULT_SETTINGS } });
      import('./api.js').then(({ api }) => api.settings.update(DEFAULT_SETTINGS)).catch((error) => {
        console.warn('Failed to persist reset settings:', error);
      });
      loadSettings().catch(() => {});
      utils.showNotification('Settings reset to defaults', 'info');
    }
  });
}

function updateTimerSettingVisibility(enabled) {
  const timerDurationSetting = document.getElementById('timerDurationSetting');
  if (timerDurationSetting) {
    timerDurationSetting.style.opacity = enabled ? '1' : '0.5';
    timerDurationSetting.style.pointerEvents = enabled ? 'auto' : 'none';
  }
}

function updateEmotionSettingsDependency(requireCamera, enableEmotionFeedback) {
  const emotionFeedback = document.getElementById('enableEmotionFeedback');
  const detectionFrequency = document.getElementById('detectionFrequency');
  const hint = document.getElementById('cameraRequirementHint');
  if (!emotionFeedback || !detectionFrequency) return;

  const cameraEnabled = Boolean(requireCamera);
  const feedbackEnabled = Boolean(enableEmotionFeedback);

  emotionFeedback.disabled = !cameraEnabled;
  detectionFrequency.disabled = !cameraEnabled || !feedbackEnabled;

  const emotionBlock = emotionFeedback.closest('.mb-3');
  const frequencyBlock = detectionFrequency.closest('.mb-3');
  if (emotionBlock) emotionBlock.style.opacity = cameraEnabled ? '1' : '0.55';
  if (frequencyBlock) frequencyBlock.style.opacity = (cameraEnabled && feedbackEnabled) ? '1' : '0.55';

  if (hint) {
    if (!cameraEnabled) {
      hint.textContent = 'Practice camera requirement is off. Assigned mode still enforces camera and emotion tracking.';
    } else if (!feedbackEnabled) {
      hint.textContent = 'Detection frequency is disabled until emotion-based feedback is enabled.';
    } else {
      hint.textContent = 'Emotion feedback options are active for practice mode when camera requirement is on.';
    }
  }
}

// Question Timer Functions
let questionTimerInterval = null;
let questionTimeRemaining = 0;
let activeQuestionTimerKey = null;

// Global Audio Context for sound effects (reuse to avoid suspension)
let globalAudioContext = null;

function getAudioContext() {
  if (!globalAudioContext) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) {
      globalAudioContext = new AudioContext();
      console.log('🎵 Audio context created');
    }
  }
  
  // Resume if suspended
  if (globalAudioContext && globalAudioContext.state === 'suspended') {
    globalAudioContext.resume().then(() => {
      console.log('🔊 Audio context resumed from suspended state');
    });
  }
  
  return globalAudioContext;
}

function startQuestionTimer(questionKey = null) {
  const settings = getEffectiveSettings();

  // Always clear any stale interval before evaluating current setting.
  stopQuestionTimer();
  
  if (!settings.enableTimer) return;

  const durationSeconds = parseInt(settings.timerDuration, 10);
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return;
  
  activeQuestionTimerKey = questionKey || `${state.currentQuestionIndex}-${Date.now()}`;
  questionTimeRemaining = durationSeconds;
  updateTimerDisplay();
  
  // Show timer UI
  const timerEl = document.getElementById('questionTimer');
  if (timerEl) {
    timerEl.style.display = 'flex';
    timerEl.style.visibility = 'visible';
    timerEl.style.opacity = '1';
    timerEl.classList.remove('timer-warning', 'timer-danger');
    console.log('👁️ Timer element shown:', {
      display: timerEl.style.display,
      visibility: timerEl.style.visibility,
      parent: timerEl.parentElement?.className
    });
  } else {
    console.error('❌ Timer element not found!');
  }
  
  console.log('✅ Timer started:', questionTimeRemaining, 'seconds for key:', activeQuestionTimerKey);
  
  questionTimerInterval = setInterval(() => {
    if (!activeQuestionTimerKey) {
      stopQuestionTimer();
      return;
    }

    questionTimeRemaining--;
    updateTimerDisplay();
    
    // Warning at 25% time remaining
    if (questionTimeRemaining <= durationSeconds * 0.25 && questionTimeRemaining > durationSeconds * 0.1) {
      if (timerEl) timerEl.classList.add('timer-warning');
    }
    
    // Danger at 10% time remaining
    if (questionTimeRemaining <= durationSeconds * 0.1) {
      if (timerEl) {
        timerEl.classList.remove('timer-warning');
        timerEl.classList.add('timer-danger');
      }
    }
    
    // Time's up
    if (questionTimeRemaining <= 0) {
      stopQuestionTimer();
      handleTimerExpired(settings.autoSubmit);
    }
  }, 1000);
}

function stopQuestionTimer() {
  if (questionTimerInterval) {
    clearInterval(questionTimerInterval);
    questionTimerInterval = null;
  }
  activeQuestionTimerKey = null;
  
  const timerEl = document.getElementById('questionTimer');
  if (timerEl) {
    timerEl.style.display = 'none';
    timerEl.classList.remove('timer-warning', 'timer-danger');
  }
}

// Simple sound effect player
function playSound(type) {
  try {
    console.log('🔊 Playing sound:', type);
    
    const audioContext = getAudioContext();
    if (!audioContext) {
      console.warn('⚠️ Web Audio API not supported');
      return;
    }
    
    // Wait for context to resume if needed
    if (audioContext.state === 'suspended') {
      console.log('⏳ Waiting for audio context to resume...');
      audioContext.resume().then(() => {
        console.log('✅ Audio context active, playing sound');
        playTone(audioContext, type);
      }).catch(err => {
        console.warn('❌ Failed to resume audio context:', err);
      });
    } else {
      playTone(audioContext, type);
    }
  } catch (error) {
    console.warn('⚠️ Sound playback failed:', error.message);
  }
}

function playTone(audioContext, type) {
  try {
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    oscillator.type = 'sine';
    
    const now = audioContext.currentTime;
    
    if (type === 'correct') {
      // Happy ascending tone
      oscillator.frequency.setValueAtTime(523.25, now); // C5
      oscillator.frequency.linearRampToValueAtTime(659.25, now + 0.1); // E5
      gainNode.gain.setValueAtTime(0.3, now);
      gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
      oscillator.start(now);
      oscillator.stop(now + 0.2);
    } else {
      // Descending tone for incorrect
      oscillator.frequency.setValueAtTime(440, now); // A4
      oscillator.frequency.linearRampToValueAtTime(349.23, now + 0.15); // F4
      gainNode.gain.setValueAtTime(0.25, now);
      gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.25);
      oscillator.start(now);
      oscillator.stop(now + 0.25);
    }
    
    console.log('✅ Sound played successfully:', type);
  } catch (error) {
    console.warn('❌ playTone error:', error.message);
  }
}

function updateTimerDisplay() {
  const timerEl = document.getElementById('questionTimer');
  const timerDisplayEl = document.getElementById('questionTimerDisplay');
  if (!timerEl || !timerDisplayEl) return;
  
  const minutes = Math.floor(questionTimeRemaining / 60);
  const seconds = questionTimeRemaining % 60;
  timerDisplayEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function handleTimerExpired(autoSubmit) {
  console.log('⏰ Timer expired! Auto-submit:', autoSubmit);
  utils.showNotification('Time is up!', 'warning');
  
  if (autoSubmit) {
    console.log('🚀 Auto-submit enabled - forcing submission');
    
    // Try multiple approaches to submit
    const submitBtn = document.getElementById('submitAnswer');
    console.log('Submit button:', {
      exists: !!submitBtn,
      visible: submitBtn?.style?.display,
      disabled: submitBtn?.disabled
    });
    
    // Always call handleQuestionSubmit directly for reliability
    setTimeout(() => {
      console.log('⚡ Triggering handleQuestionSubmit directly');
      handleQuestionSubmit({ forceTimeoutSubmit: true, autoAdvanceOnTimeout: true });
    }, 100); // Small delay to ensure timer UI updates
    
  } else {
    // Just show a notification, let user submit manually
    utils.showNotification('Please submit your answer', 'info');
  }
}

// ===== SHARED =====
function updateUserInfo() {
  const nameEl = document.getElementById('userName');
  const avatarEl = document.getElementById('userAvatar');
  const userInfoEl = document.querySelector('.user-info');
  
  if (state.currentUser) {
    if (nameEl) nameEl.textContent = state.currentUser.name;
    if (avatarEl) avatarEl.textContent = state.currentUser.name.charAt(0).toUpperCase();
  }

  if (userInfoEl) {
    userInfoEl.classList.add('hydrated');
  }
}

function setupProfileMenu() {
  const menu = document.querySelector('.profile-menu');
  const toggleBtn = document.getElementById('profileMenuToggle');
  if (!menu || !toggleBtn) return;
  if (toggleBtn.dataset.bound === 'true') return;
  toggleBtn.dataset.bound = 'true';

  toggleBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const isOpen = menu.classList.toggle('open');
    toggleBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  });

  document.addEventListener('click', (e) => {
    if (!menu.contains(e.target)) {
      menu.classList.remove('open');
      toggleBtn.setAttribute('aria-expanded', 'false');
    }
  });
}

function setupLogout() {
  const btn = document.getElementById('logoutBtn');
  if (btn) {
    if (btn.dataset.bound === 'true') return;
    btn.dataset.bound = 'true';
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      if (!confirm('Are you sure you want to log out?')) return;
      auth.logout();
    });
  }
}

function setupCameraControls() {
  const startBtn = document.getElementById('startCamera');
  const stopBtn = document.getElementById('stopCamera');

  if(startBtn) {
    startBtn.addEventListener('click', async () => {
      try {
        // Disable button during initialization
        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
        
        // Load models if not loaded
        if(!state.modelsLoaded && !state.usingSimulatedEmotions) {
          await emotionDetector.loadModels();
        }
        
        // Start camera
        const cameraStarted = await emotionDetector.startCamera();
        
        // Start detection
        if (cameraStarted) {
          emotionDetector.startDetection();

          const cameraRequired = isCameraRequiredForLearning();

          if (state.usingSimulatedEmotions) {
            utils.showNotification('Camera started in simulation mode. Controls stay locked until real face detection is active.', 'warning');
          } else {
            utils.showNotification(
              cameraRequired
                ? 'Camera started. Look at the screen until face detection points appear to unlock controls.'
                : 'Camera started. Learning controls remain available because camera is optional.',
              'info'
            );
          }

          if (cameraRequired) {
            setLearningControlsEnabled(false);
            setQuestionInteractionLocked(true);
          } else {
            setLearningControlsEnabled(true);
            setQuestionInteractionLocked(false);
          }
          updateLearningStatusCard();
          updateLearningApplyButtonState();
        } else {
          utils.showNotification('Camera failed to start. Please try again.', 'error');
          startBtn.disabled = false;
          startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
        }
      } catch (error) {
        console.error('Error starting camera:', error);
        utils.showNotification('Camera error: ' + error.message, 'error');
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
      }
    });
  }

  if(stopBtn) {
    stopBtn.addEventListener('click', () => {
      const cameraRequired = isCameraRequiredForLearning();

      emotionDetector.stopCamera();

      if (cameraRequired) {
        stopQuestionTimer();
        setLearningControlsEnabled(false);
        setQuestionInteractionLocked(true);

        // Strict mode: hide questions when camera is required and stopped.
        const questionContainer = document.getElementById('questionContainer');
        if (questionContainer) {
          questionContainer.style.display = 'none';
        }
      } else {
        // Optional mode: camera toggle must not impact question/control availability.
        setLearningControlsEnabled(true);
        setQuestionInteractionLocked(false);
      }

      updateLearningStatusCard();
      updateLearningApplyButtonState();
      
      // Re-enable start button
      const startBtn = document.getElementById('startCamera');
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
      }
      
      utils.showNotification(
        cameraRequired
          ? 'Camera stopped. Start camera to continue learning.'
          : 'Camera stopped. Learning remains available because camera is optional.',
        'info'
      );
    });
  }
}

// ===== NAVIGATION GUARD (Back button logout on protected pages) =====
let backGuardInstalled = false;
function installBackNavigationGuard() {
  if (window.__elevateShellRouting) return;
  if (backGuardInstalled) return;
  backGuardInstalled = true;

  // Ensure current entry is a guard state, then push another to catch Back.
  try {
    history.replaceState({ elevateGuard: true }, document.title, window.location.href);
    history.pushState({ elevateGuard: true }, document.title, window.location.href);
  } catch (err) {
    console.warn('History guard not applied:', err);
    return;
  }

  window.addEventListener('popstate', (event) => {
    // Only act on our guard states
    if (!event.state || !event.state.elevateGuard) {
      return;
    }

    const confirmLogout = window.confirm(
      'You are leaving a protected page. You will be logged out and need to log in again. Continue?'
    );

    if (confirmLogout) {
      auth.logout();
    } else {
      // Re-arm the guard so Back again will still prompt
      history.pushState({ elevateGuard: true }, document.title, window.location.href);
    }
  });
}