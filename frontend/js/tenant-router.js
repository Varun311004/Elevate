/**
 * tenant-router.js
 * School-slug aware frontend routing.
 *
 * Rules:
 * - Never use on index.html.
 * - If the logged-in user has a school slug, URLs become /<slug>/<page>
 * - If there is no school slug, URLs stay /<page>
 * - Works only on frontend navigation; backend already serves slug routes.
 */

const STORAGE_KEY = 'elevate_school_slug';
const DEFAULT_PAGES = new Set([
  'dashboard.html',
  'learning.html',
  'reports.html',
  'profile.html',
  'settings.html',
  'teacher-dashboard.html',
  'admin.html',
]);

function normalizeSlug(value) {
  const s = String(value || '').trim().toLowerCase();
  return s || null;
}

function normalizePage(pageName) {
  return String(pageName || '').trim().replace(/^\/+/, '');
}

export const TenantRouter = {
  setSlug(slug) {
    const clean = normalizeSlug(slug);
    if (clean) sessionStorage.setItem(STORAGE_KEY, clean);
    else sessionStorage.removeItem(STORAGE_KEY);
  },

  getSlug() {
    return sessionStorage.getItem(STORAGE_KEY) || null;
  },

  clearSlug() {
    sessionStorage.removeItem(STORAGE_KEY);
  },

  buildUrl(pageName) {
    const page = normalizePage(pageName);
    const slug = this.getSlug();
    return slug ? `/${slug}/${page}` : `/${page}`;
  },

  isSluggedPath() {
    const segments = window.location.pathname.split('/').filter(Boolean);
    return segments.length >= 2 && DEFAULT_PAGES.has(segments[segments.length - 1]);
  },

  currentPageFromPath() {
    const segments = window.location.pathname.split('/').filter(Boolean);
    if (!segments.length) return 'index.html';
    const last = segments[segments.length - 1];
    return last.includes('.') ? last : 'index.html';
  },

  redirectAfterLogin(pageName, schoolSlug) {
    this.setSlug(schoolSlug);
    window.location.replace(this.buildUrl(pageName));
  },

  enforceOnPageLoad(currentPageName) {
    const page = normalizePage(currentPageName);
    if (!page || page === 'index.html') return;

    const slug = this.getSlug();
    const segments = window.location.pathname.split('/').filter(Boolean);
    const urlPage = segments.length ? segments[segments.length - 1] : '';
    const urlSlug = segments.length >= 2 ? segments[0] : null;

    if (slug) {
      if (urlPage !== page || urlSlug !== slug) {
        window.location.replace(this.buildUrl(page));
      }
      return;
    }

    if (urlPage !== page || urlSlug) {
      window.location.replace(`/${page}`);
    }
  },

  patchNavLinks(rootEl = document) {
    rootEl.querySelectorAll('a[data-page]').forEach((a) => {
      const page = a.dataset.page;
      if (page) a.setAttribute('href', this.buildUrl(page));
    });
  },

  wireButtons(rootEl = document) {
    rootEl.querySelectorAll('[data-go-page]').forEach((el) => {
      const page = el.dataset.goPage;
      if (!page) return;
      el.addEventListener('click', (e) => {
        e.preventDefault();
        window.location.href = this.buildUrl(page);
      });
    });
  },
};

window.TenantRouter = TenantRouter;