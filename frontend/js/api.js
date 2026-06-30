// API module for backend communication
import { config } from './config.js';

const SCHOOL_SLUG_HINT_KEY = 'elevate_school_slug_hint';
const SCHOOL_SLUG_HINT_COOKIE = 'elevate_school_slug_hint';

export const api = {
  buildFriendlyApiError(endpoint, method, status, serverMessage = '') {
    const route = String(endpoint || '').toLowerCase();
    const verb = String(method || 'GET').toUpperCase();
    const cleanedServerMessage = String(serverMessage || '').trim();

    if (route.includes('/auth/login')) {
      if (status === 401) return 'We could not sign you in. Please check your email and password and try again.';
      return 'Sign-in is unavailable right now. Please try again.';
    }

    if (route.includes('/auth/signup')) {
      if (status === 409) return 'This email is already registered. Please sign in or use a different email.';
      return 'We could not create your account right now. Please try again.';
    }

    if (route.includes('/teacher/tests') && verb === 'POST') {
      if (status === 408) {
        return 'Test generation is taking longer than expected. Please wait a bit and refresh the Tests section.';
      }
      return 'We could not create the test right now. Please try again.';
    }

    if (route.includes('/teacher/question-bank/generate')) {
      return 'We could not prepare questions right now. Please try again.';
    }

    if (route.includes('/teacher/documents/upload') || route.includes('/teacher/documents/cleanup')) {
      if (cleanedServerMessage) return cleanedServerMessage;
      return 'We could not process that document request. Please check file type and size and try again.';
    }

    if (route.includes('/teacher/documents/') && cleanedServerMessage && status >= 400 && status < 500) {
      return cleanedServerMessage;
    }

    if (status === 400) return cleanedServerMessage || 'We could not process that request. Please check your details and try again.';
    if (status === 401) return 'Your session has expired. Please sign in again.';
    if (status === 403) return 'You do not have permission to do that action.';
    if (status === 404) return 'The requested item could not be found.';
    if (status === 408) return cleanedServerMessage || 'Request timed out. Please try again.';
    if (status === 409) return cleanedServerMessage || 'A conflicting record already exists. Please review your input and try again.';
    if (status === 429) return cleanedServerMessage || 'Too many requests right now. Please wait a moment and try again.';
    if ([500, 502, 503, 504].includes(Number(status))) {
      return 'Something went wrong on our side. Please try again in a moment.';
    }

    if (cleanedServerMessage) {
      return cleanedServerMessage;
    }

    return 'We could not complete your request. Please try again.';
  },

  buildFriendlyNetworkError() {
    return 'We could not connect to the server right now. Please check your connection and try again.';
  },

  // Get stored JWT token from both localStorage and sessionStorage
  getToken() {
    try {
      // Try localStorage first (Remember Me = checked)
      let session = localStorage.getItem('elevate_user_session');
      let storage = 'localStorage';
      
      // If not in localStorage, try sessionStorage (Remember Me = unchecked)
      if (!session) {
        session = sessionStorage.getItem('elevate_user_session');
        storage = 'sessionStorage';
      }
      
      if (session) {
        const parsed = JSON.parse(session);
        console.log(`Retrieved token from ${storage}:`, parsed.token ? 'Token exists' : 'No token');
        return parsed.token || null;
      } else {
        console.log('No session found in localStorage or sessionStorage');
      }
    } catch (e) {
      console.warn('Failed to get token', e);
    }
    return null;
  },

  // Base request function with JWT support
  async request(endpoint, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    let url = `${config.API_BASE_URL}${endpoint}`;
    const token = this.getToken();
    const isFormDataPayload = typeof FormData !== 'undefined' && options.body instanceof FormData;
    const timeoutMs = Number(options.timeoutMs || config.API_REQUEST_TIMEOUT_MS || 12000);
    const timeoutController = new AbortController();
    let timeoutHandle = null;

    // Cache-bust GET requests without adding custom headers that can break CORS preflight.
    if (method === 'GET') {
      const separator = url.includes('?') ? '&' : '?';
      url = `${url}${separator}_ts=${Date.now()}`;
    }
    
    const defaultOptions = {
      headers: {
        ...(isFormDataPayload ? {} : { 'Content-Type': 'application/json' }),
        ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
      },
      cache: 'no-store',
      signal: options.signal || timeoutController.signal,
      ...options,
    };

    // Merge headers if provided in options
    if (options.headers) {
      defaultOptions.headers = {
        ...defaultOptions.headers,
        ...options.headers,
      };
    }

    try {
      timeoutHandle = setTimeout(() => timeoutController.abort(), timeoutMs);
      const response = await fetch(url, defaultOptions);
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        // Handle 401 - token expired or invalid
        if (response.status === 401) {
          // Clear both storage locations and redirect to login
          localStorage.removeItem('elevate_user_session');
          sessionStorage.removeItem('elevate_user_session');
          localStorage.removeItem(SCHOOL_SLUG_HINT_KEY);
          sessionStorage.removeItem(SCHOOL_SLUG_HINT_KEY);
          document.cookie = `${SCHOOL_SLUG_HINT_COOKIE}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax`;
          const currentPath = window.location.pathname;
          if (currentPath !== '/index.html' && currentPath !== '/') {
            window.location.href = '/index.html';
          }
        }
        
        const serverMessage = data.error || data.message || '';
        const userMessage = this.buildFriendlyApiError(endpoint, method, response.status, serverMessage);
        const requestError = new Error(userMessage);
        requestError.status = response.status;
        requestError.payload = data;
        requestError.userMessage = userMessage;
        requestError.serverMessage = serverMessage;
        throw requestError;
      }

      return data;
    } catch (error) {
      if (error && error.name === 'AbortError') {
        const timeoutMessage = this.buildFriendlyApiError(endpoint, method, 408, '');
        const timeoutError = new Error(timeoutMessage);
        timeoutError.userMessage = timeoutMessage;
        timeoutError.serverMessage = '';
        timeoutError.status = 408;
        throw timeoutError;
      }

      if (!error.userMessage) {
        const fallback = this.buildFriendlyNetworkError();
        error.serverMessage = error.message || '';
        error.userMessage = fallback;
        error.message = fallback;
      }
      console.error('API Error:', error);
      throw error;
    } finally {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
    }
  },

  // Auth endpoints -> real Flask backend
  auth: {
    async login(email, password, schoolSlug = null) {
      const payload = { email, password };
      if (schoolSlug) payload.school_slug = schoolSlug;

      const data = await api.request('/auth/login', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      return {
        success: true,
        user: data.user,
        token: data.token,
      };
    },

    async signup(nameOrPayload, email, password, grade, role = 'student') {
      const payload = (typeof nameOrPayload === 'object' && nameOrPayload !== null)
        ? { ...nameOrPayload }
        : { name: nameOrPayload, email, password, grade, role };

      const data = await api.request('/auth/signup', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      return {
        success: true,
        user: data.user,
        token: data.token,
      };
    },

    async logout() {
      try {
        await api.request('/auth/logout', {
          method: 'POST',
        });
      } catch (error) {
        console.warn('Logout API call failed, but clearing local session anyway');
      }
      return { success: true };
    },

    async getProfile() {
      const data = await api.request('/auth/me', {
        method: 'GET',
      });
      return {
        success: true,
        user: data.user,
      };
    },
  },

  // Question endpoints
  questions: {
    async list(params = {}) {
      const queryParams = new URLSearchParams();
      if (params.grade) queryParams.append('grade', params.grade);
      if (params.subject) queryParams.append('subject', params.subject);
      if (params.topic) queryParams.append('topic', params.topic);
      if (params.difficulty) queryParams.append('difficulty', params.difficulty);
      if (params.exclude_answered !== undefined) queryParams.append('exclude_answered', params.exclude_answered);
      if (params.limit) queryParams.append('limit', params.limit);
      if (params.offset) queryParams.append('offset', params.offset);
      
      const queryString = queryParams.toString();
      const endpoint = queryString ? `/questions?${queryString}` : '/questions';
      
      return await api.request(endpoint, { method: 'GET' });
    },

    async get(id) {
      return await api.request(`/questions/${id}`, { method: 'GET' });
    },

    async submit(questionId, selectedIndex, timeSpent, emotion = null) {
      return await api.request(`/questions/${questionId}/submit`, {
        method: 'POST',
        body: JSON.stringify({
          selected_index: selectedIndex,
          time_spent: timeSpent,
          emotion: emotion
        }),
      });
    },

    async generate(params = {}) {
      const query = new URLSearchParams();
      if (params.grade) query.append('grade', params.grade);
      if (params.subject) query.append('subject', params.subject);
      if (params.topic) query.append('topic', params.topic);
      if (params.difficulty) query.append('difficulty', params.difficulty);
      if (params.count) query.append('count', params.count);
      if (params.exclude_answered !== undefined) query.append('exclude_answered', params.exclude_answered);
      const endpoint = `/questions/generate?${query.toString()}`;
      return await api.request(endpoint, { method: 'GET', timeoutMs: 45000 });
    },

    async topics(params = {}) {
      const query = new URLSearchParams();
      if (params.grade) query.append('grade', params.grade);
      if (params.subject) query.append('subject', params.subject);
      const endpoint = `/questions/topics?${query.toString()}`;
      return await api.request(endpoint, { method: 'GET' });
    }
  },

  // Progress endpoints
  progress: {
    async get() {
      return await api.request('/progress', { method: 'GET' });
    },

    async getSubject(subject) {
      return await api.request(`/progress/${subject}`, { method: 'GET' });
    },

    async getDashboardStats() {
      return await api.request('/progress/stats/dashboard', { method: 'GET' });
    }
  },

  // Emotion endpoints
  emotions: {
    async log(emotion, confidence, context = null) {
      return await api.request('/emotions', {
        method: 'POST',
        body: JSON.stringify({
          emotion: emotion,
          confidence: confidence,
          context: context
        }),
      });
    },

    async getHistory(params = {}) {
      const queryParams = new URLSearchParams();
      if (params.limit) queryParams.append('limit', params.limit);
      if (params.offset) queryParams.append('offset', params.offset);
      if (params.days) queryParams.append('days', params.days);
      if (params.context) queryParams.append('context', params.context);
      
      const queryString = queryParams.toString();
      const endpoint = queryString ? `/emotions/history?${queryString}` : '/emotions/history';
      
      return await api.request(endpoint, { method: 'GET' });
    },

    async getSummary(days = 7) {
      return await api.request(`/emotions/summary?days=${days}`, { method: 'GET' });
    },

    async getTimeline(days = 7, groupBy = 'day') {
      return await api.request(`/emotions/timeline?days=${days}&group_by=${groupBy}`, { method: 'GET' });
    }
  },

  // Reports endpoints
  reports: {
    async getSummary(days = 30) {
      return await api.request(`/reports/summary?days=${days}`, { method: 'GET' });
    },

    async getSubjects(days = 30) {
      return await api.request(`/reports/subjects?days=${days}`, { method: 'GET' });
    },

    async getEmotions(days = 30) {
      return await api.request(`/reports/emotions?days=${days}`, { method: 'GET' });
    },

    async getTimeline(days = 30) {
      return await api.request(`/reports/timeline?days=${days}`, { method: 'GET' });
    },

    async getIntegrity(days = 30) {
      return await api.request(`/reports/integrity?days=${days}`, { method: 'GET' });
    }
  },

  // Settings endpoints
  settings: {
    async get() {
      return await api.request('/settings', { method: 'GET' });
    },

    async update(settings) {
      return await api.request('/settings', {
        method: 'PUT',
        body: JSON.stringify(settings || {}),
      });
    }
  },

  // Teacher endpoints
  teacher: {
    async getDashboard(days = 30) {
      return await api.request(`/teacher/dashboard?days=${days}&include_at_risk=0`, {
        method: 'GET',
        timeoutMs: 45000,
      });
    },

    async getTests() {
      return await api.request('/teacher/tests', { method: 'GET' });
    },

    async getTest(testId) {
      return await api.request(`/teacher/tests/${testId}`, { method: 'GET' });
    },

    async createTest(payload) {
      const timeoutMs = Number(config.API_TEST_CREATE_TIMEOUT_MS || 180000);
      return await api.request('/teacher/tests', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
        timeoutMs,
      });
    },

    async updateTest(testId, payload) {
      return await api.request(`/teacher/tests/${testId}`, {
        method: 'PUT',
        body: JSON.stringify(payload || {}),
      });
    },

    async deleteTest(testId) {
      return await api.request(`/teacher/tests/${testId}`, {
        method: 'DELETE',
      });
    },

    async getStudents(grade = '') {
      const query = grade ? `?grade=${encodeURIComponent(grade)}` : '';
      return await api.request(`/teacher/students${query}`, { method: 'GET' });
    },

    async getClassrooms() {
      return await api.request('/teacher/classrooms', { method: 'GET' });
    },

    async createClassroom(payload) {
      return await api.request('/teacher/classrooms', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async enrollClassroomByGrade(classroomId) {
      return await api.request(`/teacher/classrooms/${classroomId}/enroll-grade`, {
        method: 'POST',
      });
    },

    async addStudentToClassroom(classroomId, studentId) {
      return await api.request(`/teacher/classrooms/${classroomId}/students`, {
        method: 'POST',
        body: JSON.stringify({ student_id: studentId }),
      });
    },

    async removeStudentFromClassroom(classroomId, studentId) {
      return await api.request(`/teacher/classrooms/${classroomId}/students/${studentId}`, {
        method: 'DELETE',
      });
    },

    async getAssignments(limit = 200) {
      return await api.request(`/teacher/assignments?limit=${limit}`, { method: 'GET' });
    },

    async getDocuments() {
      return await api.request('/teacher/documents', { method: 'GET' });
    },

    async uploadDocument(file, options = {}) {
      const formData = new FormData();
      formData.append('file', file);

      if (options.title) formData.append('title', String(options.title));
      if (options.chunk_size) formData.append('chunk_size', String(options.chunk_size));
      if (options.overlap) formData.append('overlap', String(options.overlap));
      if (options.vector_store) formData.append('vector_store', String(options.vector_store));
      if (options.document_storage) formData.append('document_storage', String(options.document_storage));
      if (options.async_ingestion !== undefined) formData.append('async_ingestion', options.async_ingestion ? '1' : '0');

      return await api.request('/teacher/documents/upload', {
        method: 'POST',
        body: formData,
      });
    },

    async deleteDocument(documentId) {
      return await api.request(`/teacher/documents/${documentId}`, {
        method: 'DELETE',
      });
    },

    async cleanupDocuments(payload = {}) {
      return await api.request('/teacher/documents/cleanup', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async createAssignment(payload) {
      return await api.request('/teacher/assignments', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async updateAssignment(assignmentId, payload) {
      return await api.request(`/teacher/assignments/${assignmentId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload || {}),
      });
    },

    async deleteAssignment(assignmentId) {
      return await api.request(`/teacher/assignments/${assignmentId}`, {
        method: 'DELETE',
      });
    },

    async getReports(params = {}) {
      const query = new URLSearchParams();
      if (params.subject) query.append('subject', params.subject);
      if (params.grade) query.append('grade', params.grade);
      if (params.days) query.append('days', String(params.days));
      if (params.limit) query.append('limit', String(params.limit));
      const qs = query.toString();
      return await api.request(`/teacher/reports${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async getAnalytics(params = {}) {
      const query = new URLSearchParams();
      if (params.grade) query.append('grade', params.grade);
      if (params.days) query.append('days', String(params.days));
      const qs = query.toString();
      return await api.request(`/teacher/analytics${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async getInterventions(params = {}) {
      const query = new URLSearchParams();
      if (params.status) query.append('status', params.status);
      if (params.limit) query.append('limit', String(params.limit));
      const qs = query.toString();
      return await api.request(`/teacher/interventions${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async getRagObservability(params = {}) {
      const query = new URLSearchParams();
      if (params.days) query.append('days', String(params.days));
      if (params.limit) query.append('limit', String(params.limit));
      const qs = query.toString();
      return await api.request(`/teacher/rag/observability${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async createIntervention(payload) {
      return await api.request('/teacher/interventions', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async updateIntervention(interventionId, payload) {
      return await api.request(`/teacher/interventions/${interventionId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload || {}),
      });
    },

    async assignRemedialTest(payload) {
      return await api.request('/teacher/interventions/actions/remedial-assignment', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async createFocusedPractice(payload) {
      return await api.request('/teacher/interventions/actions/focused-practice', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async scheduleFollowUpAssignment(payload) {
      return await api.request('/teacher/interventions/actions/follow-up-assignment', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async groupWeaknessClusters(payload) {
      return await api.request('/teacher/interventions/actions/weakness-clusters', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async generateQuestionBank(payload) {
      return await api.request('/teacher/question-bank/generate', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    }
  },

  // Admin endpoints
  admin: {
    async fetchText(endpoint, method = 'GET') {
      const token = api.getToken();
      const url = `${config.API_BASE_URL}${endpoint}${endpoint.includes('?') ? '&' : '?'}_ts=${Date.now()}`;
      const response = await fetch(url, {
        method,
        headers: {
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        cache: 'no-store',
      });

      if (!response.ok) {
        const payload = await response.text().catch(() => '');
        const userMessage = api.buildFriendlyApiError(endpoint, method, response.status, payload);
        const requestError = new Error(userMessage);
        requestError.status = response.status;
        requestError.serverMessage = payload;
        requestError.userMessage = userMessage;
        throw requestError;
      }

      return await response.text();
    },

    async getStats() {
      return await api.request('/admin/stats', { method: 'GET' });
    },

    async listUsers(params = {}) {
      const query = new URLSearchParams();
      if (params.page) query.append('page', String(params.page));
      if (params.per_page) query.append('per_page', String(params.per_page));
      if (params.search) query.append('search', String(params.search));
      if (params.role) query.append('role', String(params.role));
      if (params.status) query.append('status', String(params.status));
      if (params.school_id !== undefined && params.school_id !== null && String(params.school_id) !== '') {
        query.append('school_id', String(params.school_id));
      }
      const qs = query.toString();
      return await api.request(`/admin/users${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async updateUser(userId, payload = {}) {
      return await api.request(`/admin/users/${userId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload || {}),
      });
    },

    async disableUser(userId, reason = 'Admin action') {
      return await api.request(`/admin/users/${userId}/disable`, {
        method: 'POST',
        body: JSON.stringify({ reason }),
      });
    },

    async enableUser(userId) {
      return await api.request(`/admin/users/${userId}/enable`, {
        method: 'POST',
      });
    },

    async getSchoolsHierarchy() {
      return await api.request('/admin/schools/hierarchy', { method: 'GET' });
    },

    async getSchoolHierarchyDetail(schoolId) {
      return await api.request(`/admin/schools/${schoolId}/hierarchy`, { method: 'GET' });
    },

    async createSchool(payload = {}) {
      return await api.request('/admin/schools', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async deleteSchool(schoolId) {
      return await api.request(`/admin/schools/${schoolId}`, {
        method: 'DELETE',
      });
    },

    async listTestResults(params = {}) {
      const query = new URLSearchParams();
      if (params.page) query.append('page', String(params.page));
      if (params.per_page) query.append('per_page', String(params.per_page));
      if (params.subject) query.append('subject', params.subject);
      if (params.email) query.append('email', params.email);
      if (params.start) query.append('start', params.start);
      if (params.end) query.append('end', params.end);
      if (params.status) query.append('status', params.status);
      if (params.school_id !== undefined && params.school_id !== null && String(params.school_id) !== '') {
        query.append('school_id', String(params.school_id));
      }
      if (Number.isFinite(Number(params.min_score))) query.append('min_score', String(params.min_score));
      if (Number.isFinite(Number(params.max_score))) query.append('max_score', String(params.max_score));
      const qs = query.toString();
      return await api.request(`/admin/test-results${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async getTestResultHistory(userId) {
      return await api.request(`/admin/test-results/${userId}/history`, { method: 'GET' });
    },

    async getTestResultDetail(testResultId) {
      return await api.request(`/admin/test-results/${testResultId}`, { method: 'GET' });
    },

    async exportTestResultsCsv(params = {}) {
      const query = new URLSearchParams();
      query.append('format', 'csv');
      if (params.subject) query.append('subject', params.subject);
      if (params.email) query.append('email', params.email);
      if (params.start) query.append('start', params.start);
      if (params.end) query.append('end', params.end);
      if (params.status) query.append('status', params.status);
      if (params.school_id !== undefined && params.school_id !== null && String(params.school_id) !== '') {
        query.append('school_id', String(params.school_id));
      }
      if (Number.isFinite(Number(params.min_score))) query.append('min_score', String(params.min_score));
      if (Number.isFinite(Number(params.max_score))) query.append('max_score', String(params.max_score));
      const endpoint = `/admin/test-results?${query.toString()}`;
      return await this.fetchText(endpoint, 'GET');
    },

    async triggerTraining() {
      // HF Spaces cold start + proxy can exceed normal API timeout; match backend HF client (~300s).
      return await api.request('/admin/ml/train-strict', {
        method: 'POST',
        timeoutMs: 360000,
      });
    },

    async getTrainingStatus(jobId) {
      return await api.request(`/admin/ml/train-strict/${jobId}`, {
        method: 'GET',
      });
    },

    async listTrainingJobs(params = {}) {
      const query = new URLSearchParams();
      if (params.page) query.append('page', String(params.page));
      if (params.per_page) query.append('per_page', String(params.per_page));
      if (params.status) query.append('status', String(params.status));
      const qs = query.toString();
      return await api.request(`/admin/ml/training-jobs${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async getTrainingJob(jobDbId, params = {}) {
      return await api.request(`/admin/ml/training-jobs/${jobDbId}`, { method: 'GET' });
    },

    async listModelVersions(params = {}) {
      const query = new URLSearchParams();
      if (params.page) query.append('page', String(params.page));
      if (params.per_page) query.append('per_page', String(params.per_page));
      if (params.model_name) query.append('model_name', String(params.model_name));
      const qs = query.toString();
      return await api.request(`/admin/ml/model-versions${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async getModelRegistrySummary(params = {}) {
      const query = new URLSearchParams();
      if (params.model_name) query.append('model_name', String(params.model_name));
      const qs = query.toString();
      return await api.request(`/admin/ml/model-versions/registry-summary${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async createModelVersion(payload = {}) {
      return await api.request('/admin/ml/model-versions', {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async promoteModelVersion(versionId) {
      return await api.request(`/admin/ml/model-versions/${versionId}/promote`, {
        method: 'POST',
      });
    },

    async setRollbackTarget(versionId) {
      return await api.request(`/admin/ml/model-versions/${versionId}/rollback`, {
        method: 'POST',
      });
    },

    async getMcqObservability(params = {}) {
      const query = new URLSearchParams();
      if (params.days) query.append('days', String(params.days));
      if (params.subject) query.append('subject', String(params.subject));
      const qs = query.toString();
      return await api.request(`/admin/mcq/observability${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async listAuditLogs(params = {}) {
      const query = new URLSearchParams();
      if (params.page) query.append('page', String(params.page));
      if (params.per_page) query.append('per_page', String(params.per_page));
      if (params.action) query.append('action', String(params.action));
      if (params.target_type) query.append('target_type', String(params.target_type));
      if (params.actor_id !== undefined && params.actor_id !== null && String(params.actor_id) !== '') {
        query.append('actor_id', String(params.actor_id));
      }
      if (params.date_from) query.append('date_from', String(params.date_from));
      if (params.date_to) query.append('date_to', String(params.date_to));
      const qs = query.toString();
      return await api.request(`/admin/audit-logs${qs ? `?${qs}` : ''}`, { method: 'GET' });
    },

    async exportAuditLogs(params = {}) {
      const query = new URLSearchParams();
      if (params.action) query.append('action', String(params.action));
      if (params.target_type) query.append('target_type', String(params.target_type));
      if (params.date_from) query.append('date_from', String(params.date_from));
      if (params.date_to) query.append('date_to', String(params.date_to));
      const qs = query.toString();
      return await this.fetchText(`/admin/audit-logs/export${qs ? `?${qs}` : ''}`, 'GET');
    },

    downloadCsvTemplate: () => fetchWithAuth('/api/admin/users/csv-template', {}, false), 
    bulkImportUsers: async (formData) => {
      return await api.request('/admin/users/bulk-import', {
        method: 'POST',
        body: formData,
      });
    },

    singleAddUser: async (data) => {
        return await api.request('/admin/users/single-add', { 
            method: 'POST', 
            body: JSON.stringify(data) 
        });
    },
  },

  student: {
    async getAssignedTests() {
      return await api.request('/student/assigned-tests', { method: 'GET' });
    },

    async getAvailableTests() {
      return await api.request('/student/tests', { method: 'GET' });
    },

    async startTest(testId) {
      return await api.request(`/student/tests/${testId}/start`, { method: 'POST' });
    },

    async getTestQuestions(testId) {
      return await api.request(`/student/tests/${testId}/questions`, { method: 'GET' });
    },

    async submitTestAnswer(testId, payload) {
      return await api.request(`/student/tests/${testId}/answer`, {
        method: 'POST',
        body: JSON.stringify(payload || {}),
      });
    },

    async finishTest(testId) {
      return await api.request(`/student/tests/${testId}/finish`, { method: 'POST' });
    }
  },
  
  async generateQuestions(payload) {
        const token = localStorage.getItem('elevate_token');
        const response = await fetch('/api/teacher/question-bank/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
};
