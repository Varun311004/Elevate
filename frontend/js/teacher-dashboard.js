import { auth } from './auth.js';
import { api } from './api.js';
import { utils } from './utils.js';

const charts = {};
const teacherCache = {
  tests: [],
  students: [],
  classrooms: [],
  assignments: [],
  reportRows: [],
  interventions: [],
  documents: [],
  ragObservability: null,
};

let activeAssessmentMode = 'topic';
let activeWorkspaceTarget = 'assessmentsWorkspace';
let assessmentsRealtimeIntervalId = null;
let assessmentsRealtimeInFlight = false;

const ASSESSMENTS_REALTIME_REFRESH_MS = 12000;

const OPTION_LETTERS = ['A', 'B', 'C', 'D'];

const WORKSPACE_SECTION_MAP = {
  assessmentsWorkspace: ['overviewSection', 'createSection', 'testsSection', 'assignmentsSection'],
  classroomsWorkspace: ['classroomsSection', 'studentsSection'],
  insightsWorkspace: ['reportsSection', 'analyticsSection', 'interventionsSection'],
};

const TEACHER_MESSAGE_KEYS = {
  testCreateSuccess: 'teacher.test_create_success',
  testCreateFailed: 'teacher.test_create_failed',
  testUpdateSuccess: 'teacher.test_update_success',
  testUpdateFailed: 'teacher.test_update_failed',
  testDeleteSuccess: 'teacher.test_delete_success',
  testDeleteFailed: 'teacher.test_delete_failed',
  classroomCreateSuccess: 'teacher.classroom_create_success',
  classroomCreateFailed: 'teacher.classroom_create_failed',
  assignmentCreateSuccess: 'teacher.assignment_create_success',
  assignmentCreateFailed: 'teacher.assignment_create_failed',
  studentAddSuccess: 'teacher.student_add_success',
  studentAddFailed: 'teacher.student_add_failed',
  studentRemoveSuccess: 'teacher.student_remove_success',
  studentRemoveFailed: 'teacher.student_remove_failed',
  documentUploadSuccess: 'teacher.document_upload_success',
  documentUploadFailed: 'teacher.document_upload_failed',
  documentDeleteSuccess: 'teacher.document_delete_success',
  documentDeleteFailed: 'teacher.document_delete_failed',
};

function notifyTeacherSuccess(messageKey, replacements = {}) {
  const key = TEACHER_MESSAGE_KEYS[messageKey] || 'generic.completed';
  utils.notifySuccess(key, replacements);
}

function notifyTeacherError(messageKey, error) {
  const key = TEACHER_MESSAGE_KEYS[messageKey] || 'generic.failed';
  const fallback = utils.getMessage(key);
  const message = error?.userMessage || error?.message || fallback;
  utils.showNotification(message, 'error');
}

function getKnownStudents() {
  const byId = new Map();

  const addStudent = (student) => {
    if (!student || !student.id) return;
    byId.set(Number(student.id), {
      id: Number(student.id),
      name: student.name || `Student #${student.id}`,
      email: student.email || '',
      grade: student.grade || '',
      attempts: Number(student.attempts || 0),
      avg_score: Number(student.avg_score || 0),
    });
  };

  (teacherCache.students || []).forEach(addStudent);
  (teacherCache.classrooms || []).forEach(classroom => {
    (classroom.students || []).forEach(addStudent);
  });
  (teacherCache.assignments || []).forEach(assignment => {
    if (assignment && assignment.student) {
      addStudent(assignment.student);
    }
  });

  return Array.from(byId.values()).sort((a, b) => String(a.name).localeCompare(String(b.name)));
}

const STEM_SUBJECT_TOPICS = {
  science: ['physics', 'chemistry', 'biology', 'earth science', 'environment'],
  technology: ['computer science', 'programming', 'data', 'networks', 'ai fundamentals'],
  engineering: ['mechanics', 'circuits', 'design process', 'materials', 'robotics'],
  mathematics: ['algebra', 'arithmetic', 'geometry', 'trigonometry', 'calculus'],
};

function getSession() {
  return auth.loadSession();
}

function normalizeSchoolSlug(slug) {
  return null;
}

function getCurrentPathSlug() {
  return null;
}

function roleHomePath(role, schoolSlug = null) {
  const normalizedRole = String(role || 'student').trim().toLowerCase();
  const page = normalizedRole === 'teacher'
    ? 'teacher-dashboard.html'
    : normalizedRole === 'admin'
      ? 'admin.html'
      : 'dashboard.html';
  return page;
}

function ensureTeacherSession() {
  const session = getSession();
  if (!session || !session.user || !session.token) {
    window.location.replace('/index.html');
    return null;
  }

  const role = String(session.user.role || 'student').trim().toLowerCase();
  if (role !== 'teacher') {
    const destination = roleHomePath(role, session.user.school_slug);
    window.location.replace(destination);
    return null;
  }

  return session;
}

function setBusy(button, busyText) {
  if (!button) return;
  if (!button.dataset.defaultText) {
    button.dataset.defaultText = button.innerHTML;
  }
  button.disabled = true;
  button.innerHTML = `<i class="fas fa-spinner fa-spin me-1"></i>${busyText}`;
}

function clearBusy(button) {
  if (!button) return;
  button.disabled = false;
  button.innerHTML = button.dataset.defaultText || button.innerHTML;
}

function setBusyWithElapsed(button, busyText) {
  if (!button) {
    return () => {};
  }

  setBusy(button, `${busyText} (0s)`);
  const startedAt = Date.now();
  const timerId = window.setInterval(() => {
    if (!button || !button.disabled) return;
    const elapsed = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
    button.innerHTML = `<i class="fas fa-spinner fa-spin me-1"></i>${busyText} (${elapsed}s)`;
  }, 1000);

  return () => {
    window.clearInterval(timerId);
  };
}

function formatDate(value) {
  if (!value) return '-';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '-';
  return d.toLocaleString();
}

function formatGradeDisplay(value, fallback = '-') {
  if (!value) return fallback;
  return utils.getGradeDisplayName(String(value)) || fallback;
}

function formatPercent(value) {
  return `${Number(value || 0).toFixed(1)}%`;
}

function formatGenerationDiagnostics(generationStatus) {
  const status = generationStatus && typeof generationStatus === 'object' ? generationStatus : {};
  const serviceStatus = status.service_status_code ?? '-';
  const latencyValue = Number(status.service_latency_ms);
  const latencyLabel = Number.isFinite(latencyValue) && latencyValue >= 0 ? `${latencyValue}ms` : 'n/a';
  const serviceGenerated = Number(status.service_generated_count || 0);
  const technicalSamples = Number(status.technical_sample_count || 0);

  const parts = [
    `HTTP ${serviceStatus}`,
    `Latency ${latencyLabel}`,
    `Service ${serviceGenerated}`,
  ];
  if (technicalSamples > 0) parts.push(`Samples ${technicalSamples}`);

  return {
    summary: parts.join(' | '),
    endpoint: status.service_endpoint ? String(status.service_endpoint) : '',
    error: status.service_error ? String(status.service_error) : '',
  };
}

function toTitleCase(value) {
  return String(value || '-')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, char => char.toUpperCase());
}

function toDateTimeLocalValue(value) {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '';

  const pad = n => String(n).padStart(2, '0');
  const yyyy = d.getFullYear();
  const mm = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  const hh = pad(d.getHours());
  const mi = pad(d.getMinutes());
  return `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
}

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function collectEditableQuestionsFromPanel(panel) {
  if (!panel) return [];
  const rows = Array.from(panel.querySelectorAll('.test-question-editor-row'));
  return rows.map((row, index) => {
    const id = Number(row.dataset.questionId || 0);
    const text = row.querySelector('.q-text')?.value?.trim() || '';
    const order = Number(row.querySelector('.q-order')?.value || (index + 1));
    const points = Number(row.querySelector('.q-points')?.value || 1);
    const explanation = row.querySelector('.q-explanation')?.value?.trim() || ''; // Added

    const options = Array.from(row.querySelectorAll('.q-option-input'))
      .map(input => input?.value?.trim() || '')
      .slice(0, 4);
    const correctIndex = Number(row.querySelector('.q-correct')?.value || 0);

    return {
      id,
      text,
      order,
      points,
      options,
      correct_index: correctIndex,
      explanation // Added
    };
  });
}

function getOptionRows(questionRow) {
  return Array.from(questionRow.querySelectorAll('.q-option-row'));
}

function syncCorrectAnswerDropdown(questionRow, preserveAnswerByText = false, preferredAnswerText = '') {
  if (!questionRow) return;

  const dropdown = questionRow.querySelector('.q-correct');
  if (!dropdown) return;

  const optionRows = getOptionRows(questionRow);
  if (optionRows.length === 0) return;

  const previousIndex = Number(dropdown.value || 0);
  const previousSelectedText = preserveAnswerByText
    ? (preferredAnswerText || optionRows[previousIndex]?.querySelector('.q-option-input')?.value?.trim() || '')
    : '';

  optionRows.forEach((row, idx) => {
    row.dataset.optionIndex = String(idx);
    const label = row.querySelector('.option-label');
    if (label) {
      label.textContent = OPTION_LETTERS[idx] || String(idx + 1);
    }
  });

  dropdown.innerHTML = '';
  optionRows.forEach((row, idx) => {
    const rawText = row.querySelector('.q-option-input')?.value?.trim() || '';
    const preview = rawText.length > 54 ? `${rawText.slice(0, 51)}...` : rawText;
    const option = document.createElement('option');
    option.value = String(idx);
    option.textContent = `${OPTION_LETTERS[idx] || idx + 1}. ${preview || `Option ${idx + 1}`}`;
    dropdown.appendChild(option);
  });

  let nextIndex = Number.isFinite(previousIndex) ? previousIndex : 0;
  if (preserveAnswerByText && previousSelectedText) {
    const matchIndex = optionRows.findIndex((row) => {
      const currentText = row.querySelector('.q-option-input')?.value?.trim() || '';
      return currentText === previousSelectedText;
    });
    if (matchIndex >= 0) {
      nextIndex = matchIndex;
    }
  }

  if (nextIndex < 0 || nextIndex >= optionRows.length) {
    nextIndex = 0;
  }
  dropdown.value = String(nextIndex);
}

function enableOptionRowDragAndDrop(questionRow) {
  if (!questionRow) return;
  if (questionRow.dataset.optionDragBound === '1') return;
  questionRow.dataset.optionDragBound = '1';

  const optionRows = getOptionRows(questionRow);
  let draggedRow = null;

  optionRows.forEach((row) => {
    row.setAttribute('draggable', 'true');

    row.addEventListener('dragstart', (evt) => {
      const fromHandle = evt.target?.closest?.('.option-drag-handle');
      if (!fromHandle) {
        evt.preventDefault();
        return;
      }

      const dropdown = questionRow.querySelector('.q-correct');
      const selectedIndex = Number(dropdown?.value || 0);
      const selectedText = getOptionRows(questionRow)[selectedIndex]?.querySelector('.q-option-input')?.value?.trim() || '';
      questionRow.dataset.selectedAnswerTextBeforeDrag = selectedText;

      draggedRow = row;
      row.classList.add('is-option-dragging');
      if (evt.dataTransfer) {
        evt.dataTransfer.effectAllowed = 'move';
        evt.dataTransfer.setData('text/plain', row.dataset.optionIndex || '');
      }
    });

    row.addEventListener('dragend', () => {
      row.classList.remove('is-option-dragging');
      draggedRow = null;
      getOptionRows(questionRow).forEach((el) => {
        el.classList.remove('option-drop-before', 'option-drop-after');
      });
      const selectedText = questionRow.dataset.selectedAnswerTextBeforeDrag || '';
      syncCorrectAnswerDropdown(questionRow, true, selectedText);
      delete questionRow.dataset.selectedAnswerTextBeforeDrag;
    });

    row.addEventListener('dragover', (evt) => {
      evt.preventDefault();
      if (!draggedRow || draggedRow === row) return;

      const rect = row.getBoundingClientRect();
      const insertAfter = evt.clientY > rect.top + rect.height / 2;

      row.classList.toggle('option-drop-before', !insertAfter);
      row.classList.toggle('option-drop-after', insertAfter);

      const parent = row.parentElement;
      if (!parent) return;
      if (insertAfter) {
        parent.insertBefore(draggedRow, row.nextSibling);
      } else {
        parent.insertBefore(draggedRow, row);
      }
    });

    row.addEventListener('dragleave', () => {
      row.classList.remove('option-drop-before', 'option-drop-after');
    });
  });
}

function initializeOptionEditors(panel) {
  if (!panel) return;
  const rows = Array.from(panel.querySelectorAll('.test-question-editor-row'));

  rows.forEach((questionRow) => {
    enableOptionRowDragAndDrop(questionRow);
    syncCorrectAnswerDropdown(questionRow, false);

    questionRow.querySelectorAll('.q-option-input').forEach((input) => {
      input.addEventListener('input', () => {
        syncCorrectAnswerDropdown(questionRow, false);
      });
    });
  });
}

function renumberQuestionCards(panel) {
  const cards = Array.from(panel.querySelectorAll('.test-question-editor-row'));
  cards.forEach((card, index) => {
    const order = index + 1;
    const serialEl = card.querySelector('.question-serial-number');
    const orderInput = card.querySelector('.q-order');
    if (serialEl) serialEl.textContent = String(order);
    if (orderInput) orderInput.value = String(order);
  });
}

function enableQuestionCardDragAndDrop(panel) {
  if (!panel) return;
  const list = panel.querySelector('.test-question-editor-list');
  if (!list) return;

  let draggedCard = null;
  const cards = Array.from(list.querySelectorAll('.test-question-editor-row'));

  cards.forEach(card => {
    card.setAttribute('draggable', 'true');

    card.addEventListener('dragstart', evt => {
      draggedCard = card;
      card.classList.add('is-dragging');
      if (evt.dataTransfer) {
        evt.dataTransfer.effectAllowed = 'move';
        evt.dataTransfer.setData('text/plain', card.dataset.questionId || '');
      }
    });

    card.addEventListener('dragend', () => {
      card.classList.remove('is-dragging');
      draggedCard = null;
      list.querySelectorAll('.drop-target-before, .drop-target-after').forEach(el => {
        el.classList.remove('drop-target-before', 'drop-target-after');
      });
      renumberQuestionCards(panel);
    });

    card.addEventListener('dragover', evt => {
      evt.preventDefault();
      if (!draggedCard || draggedCard === card) return;
      const rect = card.getBoundingClientRect();
      const insertAfter = evt.clientY > rect.top + rect.height / 2;
      card.classList.toggle('drop-target-before', !insertAfter);
      card.classList.toggle('drop-target-after', insertAfter);

      if (insertAfter) {
        list.insertBefore(draggedCard, card.nextSibling);
      } else {
        list.insertBefore(draggedCard, card);
      }
    });

    card.addEventListener('dragleave', () => {
      card.classList.remove('drop-target-before', 'drop-target-after');
    });
  });

  renumberQuestionCards(panel);
}

function activateSection(targetId) {
  const nav = document.getElementById('teacherNav');
  if (!nav) return;
  const contentArea = document.getElementById('teacherContentArea');

  const sectionsToShow = WORKSPACE_SECTION_MAP[targetId] || [targetId];

  document.querySelectorAll('.teacher-section').forEach(section => {
    section.classList.remove('active');
  });

  sectionsToShow.forEach(sectionId => {
    const section = document.getElementById(sectionId);
    if (section) {
      section.classList.add('active');
    }
  });

  nav.querySelectorAll('.nav-link').forEach(item => {
    const isActive = item.dataset.target === targetId;
    item.classList.toggle('active', isActive);
    item.setAttribute('aria-current', isActive ? 'page' : 'false');
  });

  if (contentArea) {
    contentArea.scrollTo({ top: 0, behavior: 'smooth' });
  }

  activeWorkspaceTarget = targetId;
  syncAssessmentsRealtimeRefreshState();
}

function stopAssessmentsRealtimeRefresh() {
  if (assessmentsRealtimeIntervalId) {
    window.clearInterval(assessmentsRealtimeIntervalId);
    assessmentsRealtimeIntervalId = null;
  }
}

async function refreshAssessmentsRealtime(options = {}) {
  const force = Boolean(options.force);
  if (!force) {
    if (activeWorkspaceTarget !== 'assessmentsWorkspace') return;
    if (document.hidden) return;
  }

  if (assessmentsRealtimeInFlight) return;
  assessmentsRealtimeInFlight = true;

  try {
    await Promise.allSettled([
      loadOverview(),
      loadTests(),
      loadAssignments(),
    ]);
    syncAssignmentFormOptions();
  } catch (error) {
    console.warn('Assessments realtime refresh failed:', error);
  } finally {
    assessmentsRealtimeInFlight = false;
  }
}

function startAssessmentsRealtimeRefresh() {
  stopAssessmentsRealtimeRefresh();
  assessmentsRealtimeIntervalId = window.setInterval(() => {
    refreshAssessmentsRealtime().catch(() => {});
  }, ASSESSMENTS_REALTIME_REFRESH_MS);
}

function syncAssessmentsRealtimeRefreshState() {
  if (activeWorkspaceTarget === 'assessmentsWorkspace') {
    startAssessmentsRealtimeRefresh();
    return;
  }
  stopAssessmentsRealtimeRefresh();
}

async function refreshSectionData(targetId) {
  try {
    if (targetId === 'assessmentsWorkspace') {
      const results = await Promise.allSettled([
        loadOverview(),
        loadTests(),
        loadAssignments(),
        loadClassrooms(),
        loadStudents(),
      ]);
      if (results.some(item => item.status === 'rejected')) {
        utils.showNotification('Some assessment data is still loading. You can continue using the page.', 'warning');
      }
      syncAssignmentFormOptions();
      return;
    }

    if (targetId === 'classroomsWorkspace') {
      const results = await Promise.allSettled([loadClassrooms(), loadStudents()]);
      if (results.some(item => item.status === 'rejected')) {
        utils.showNotification('Some classroom data could not be refreshed right now.', 'warning');
      }
      syncAssignmentFormOptions();
      return;
    }

    if (targetId === 'insightsWorkspace') {
      const results = await Promise.allSettled([loadReports(), loadAnalytics(), loadInterventions()]);
      if (results.some(item => item.status === 'rejected')) {
        utils.showNotification('Some insight panels could not be refreshed right now.', 'warning');
      }
      return;
    }

    if (targetId === 'overviewSection') {
      await loadOverview();
      return;
    }
    if (targetId === 'testsSection') {
      await loadTests();
      syncAssignmentFormOptions();
      return;
    }
    if (targetId === 'studentsSection') {
      await loadStudents();
      renderClassroomsTable();
      syncAssignmentFormOptions();
      return;
    }
    if (targetId === 'classroomsSection') {
      const results = await Promise.allSettled([loadClassrooms(), loadStudents()]);
      if (results.some(item => item.status === 'rejected')) {
        utils.showNotification('Some classroom data could not be refreshed right now.', 'warning');
      }
      syncAssignmentFormOptions();
      return;
    }
    if (targetId === 'assignmentsSection') {
      const results = await Promise.allSettled([
        loadAssignments(),
        loadTests(),
        loadClassrooms(),
        loadStudents(),
      ]);
      if (results.some(item => item.status === 'rejected')) {
        utils.showNotification('Some assignment data could not be refreshed right now.', 'warning');
      }
      syncAssignmentFormOptions();
      return;
    }
    if (targetId === 'reportsSection') {
      await loadReports();
      return;
    }
    if (targetId === 'analyticsSection') {
      await loadAnalytics();
      return;
    }
    if (targetId === 'interventionsSection') {
      await loadInterventions();
    }
  } catch (error) {
    console.error(error);
    utils.showNotification(error.message || 'Failed to refresh section data.', 'error');
  }
}

function bindSidebarNav() {
  const nav = document.getElementById('teacherNav');
  if (!nav) return;

  const resolveWorkspaceTarget = (rawTarget) => {
    const target = String(rawTarget || '').trim();
    if (!target) return 'assessmentsWorkspace';
    if (WORKSPACE_SECTION_MAP[target]) return target;

    for (const [workspaceId, sectionIds] of Object.entries(WORKSPACE_SECTION_MAP)) {
      if (sectionIds.includes(target)) return workspaceId;
    }

    return 'assessmentsWorkspace';
  };

  nav.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', evt => {
      evt.preventDefault();
      const targetId = resolveWorkspaceTarget(link.dataset.target);
      activateSection(targetId);
      window.location.hash = targetId;
      refreshSectionData(targetId).catch(() => {});
    });
  });

  const initialHash = (window.location.hash || '').replace('#', '');
  const initialWorkspace = resolveWorkspaceTarget(initialHash);
  if (initialWorkspace) {
    activateSection(initialWorkspace);
    window.location.hash = initialWorkspace;
    refreshSectionData(initialWorkspace).catch(() => {});
  } else {
    activateSection('assessmentsWorkspace');
    window.location.hash = 'assessmentsWorkspace';
    refreshSectionData('assessmentsWorkspace').catch(() => {});
  }
}

function setSelectOptions(selectEl, options, includeAny = false, anyLabel = 'All Subjects') {
  if (!selectEl) return;
  const normalized = Array.from(new Set((options || []).map(v => String(v || '').trim()).filter(Boolean)));
  const rows = [];
  if (includeAny) {
    rows.push('<option value="">' + anyLabel + '</option>');
  }
  normalized.forEach(value => {
    rows.push(`<option value="${escapeHtml(value)}">${escapeHtml(value.charAt(0).toUpperCase() + value.slice(1))}</option>`);
  });
  selectEl.innerHTML = rows.join('');
}

function syncReportSubjectOptions() {
  const reportSubject = document.getElementById('reportSubject');
  if (!reportSubject) return;

  const selected = reportSubject.value;
  const fromTests = (teacherCache.tests || []).map(t => String(t.subject || '').trim()).filter(Boolean);
  const fromReports = (teacherCache.reportRows || []).map(r => String(r.subject || '').trim()).filter(Boolean);
  const source = [...fromTests, ...fromReports];

  if (source.length === 0) {
    reportSubject.innerHTML = '<option value="">All Subjects</option>';
    return;
  }

  const uniq = Array.from(new Set(source));
  reportSubject.innerHTML = '<option value="">All Subjects</option>' + uniq.map(subj => {
    return `<option value="${escapeHtml(subj)}">${escapeHtml(subj)}</option>`;
  }).join('');

  if (selected && uniq.includes(selected)) {
    reportSubject.value = selected;
  }
}

async function refreshTopicDropdown() {
  const subject = document.getElementById('testSubject')?.value || '';
  const grade = document.getElementById('testGrade')?.value || '';
  const topicSelect = document.getElementById('testTopic');
  if (!topicSelect) return;
  const previous = topicSelect.value;

  let topicPool = [];
  if (subject && STEM_SUBJECT_TOPICS[subject.toLowerCase()]) {
    topicPool = topicPool.concat(STEM_SUBJECT_TOPICS[subject.toLowerCase()]);
  }

  try {
    if (subject) {
      const server = await api.questions.topics({ subject, grade });
      if (Array.isArray(server.topics)) {
        topicPool = topicPool.concat(server.topics.map(t => String(t).toLowerCase()));
      }
    }
  } catch (error) {
    console.warn('Topic fetch fallback to local STEM map only', error);
  }

  setSelectOptions(topicSelect, topicPool, false);

  const options = Array.from(topicSelect.options).map(opt => opt.value);
  if (previous && options.includes(previous)) {
    topicSelect.value = previous;
  }
  if (!topicSelect.value && topicSelect.options.length > 0) {
    topicSelect.value = topicSelect.options[0].value;
  }
}

async function initializeStemDropdowns() {
  const subjectOptions = Object.keys(STEM_SUBJECT_TOPICS);
  const testSubject = document.getElementById('testSubject');
  const ragTestSubject = document.getElementById('ragTestSubject');
  const reportSubject = document.getElementById('reportSubject');

  setSelectOptions(testSubject, subjectOptions, false);
  setSelectOptions(ragTestSubject, subjectOptions, false);
  if (reportSubject) {
    reportSubject.innerHTML = '<option value="">All Subjects</option>';
  }

  if (testSubject && !testSubject.value && subjectOptions.length > 0) {
    testSubject.value = subjectOptions[0];
  }
  if (ragTestSubject && !ragTestSubject.value && subjectOptions.length > 0) {
    ragTestSubject.value = subjectOptions[0];
  }

  await refreshTopicDropdown();

  if (testSubject) {
    testSubject.addEventListener('change', () => {
      refreshTopicDropdown().catch(() => {});
    });
  }

  const gradeSelect = document.getElementById('testGrade');
  if (gradeSelect) {
    gradeSelect.addEventListener('change', () => {
      refreshTopicDropdown().catch(() => {});
    });
  }
}

function setupProfileMenu(session) {
  const userName = document.getElementById('teacherUserName');
  const avatar = document.getElementById('teacherUserAvatar');
  const userInfo = document.getElementById('teacherUserInfo');
  const toggle = document.getElementById('profileMenuToggle');
  const menu = document.querySelector('.profile-menu');

  const name = session.user.name || 'Teacher';
  if (userName) userName.textContent = name;
  if (avatar) avatar.textContent = name.charAt(0).toUpperCase();
  if (userInfo) userInfo.classList.add('hydrated');

  if (toggle && menu) {
    toggle.addEventListener('click', evt => {
      evt.stopPropagation();
      menu.classList.toggle('open');
    });

    document.addEventListener('click', evt => {
      if (!menu.contains(evt.target)) {
        menu.classList.remove('open');
      }
    });
  }

  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      await auth.logout();
    });
  }
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

function renderSubjectChart(items) {
  const canvas = document.getElementById('teacherSubjectChart');
  if (!canvas || typeof Chart === 'undefined') return;
  destroyChart('teacherSubjectChart');

  const rows = (Array.isArray(items) ? items : []).map(item => ({
    subject: item.subject || 'Unknown',
    avg_score: Number(item.avg_score || 0),
    total_attempts: Number(item.total_attempts || 0),
  }));

  const labels = rows.map(item => toTitleCase(item.subject));
  const avgData = rows.map(item => item.avg_score);
  const attemptsData = rows.map(item => item.total_attempts);
  const palette = ['#2563eb', '#0891b2', '#16a34a', '#f59e0b', '#db2777', '#7c3aed', '#dc2626', '#0d9488'];

  charts.teacherSubjectChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          type: 'bar',
          label: 'Average Score %',
          data: avgData,
          borderWidth: 1,
          backgroundColor: labels.map((_, i) => `${palette[i % palette.length]}CC`),
          borderColor: labels.map((_, i) => palette[i % palette.length]),
          borderRadius: 8,
          yAxisID: 'y',
        },
        {
          type: 'line',
          label: 'Attempts',
          data: attemptsData,
          borderColor: '#111827',
          backgroundColor: 'rgba(17, 24, 39, 0.12)',
          tension: 0.28,
          pointRadius: 3,
          pointHoverRadius: 5,
          borderWidth: 2,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'bottom' },
        tooltip: {
          callbacks: {
            label(context) {
              if (context.dataset.label === 'Average Score %') {
                return `Average Score: ${Number(context.parsed.y || 0).toFixed(1)}%`;
              }
              return `Attempts: ${Number(context.parsed.y || 0)}`;
            }
          }
        }
      },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: value => `${value}%` }, title: { display: true, text: 'Average Score' } },
        y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Attempts' } },
      },
    },
  });
}

function buildGradePerformanceFromStudents(studentRows) {
  const rows = Array.isArray(studentRows) ? studentRows : [];
  const map = new Map();

  rows.forEach(item => {
    const grade = item.grade || 'unknown';
    const attempts = Number(item.total_attempts || 0);
    const avg = Number(item.avg_score || 0);
    const bucket = map.get(grade) || {
      grade,
      total_students: 0,
      active_students: 0,
      total_attempts: 0,
      weighted_sum: 0,
    };
    bucket.total_students += 1;
    bucket.total_attempts += attempts;
    bucket.weighted_sum += avg * attempts;
    if (attempts > 0) bucket.active_students += 1;
    map.set(grade, bucket);
  });

  const order = { elementary: 0, middle: 1, high: 2, college: 3, unknown: 9 };
  return Array.from(map.values())
    .map(item => ({
      grade: item.grade,
      total_students: item.total_students,
      active_students: item.active_students,
      total_attempts: item.total_attempts,
      avg_score: item.total_attempts > 0 ? Number((item.weighted_sum / item.total_attempts).toFixed(2)) : 0,
    }))
    .sort((a, b) => (order[a.grade] ?? 99) - (order[b.grade] ?? 99));
}

function renderGradeLevelChart(items) {
  const canvas = document.getElementById('teacherDifficultyChart');
  if (!canvas || typeof Chart === 'undefined') return;
  destroyChart('teacherDifficultyChart');

  const rows = (Array.isArray(items) ? items : []).map(item => ({
    grade: item.grade || 'unknown',
    avg_score: Number(item.avg_score || 0),
    active_students: Number(item.active_students || 0),
  }));

  charts.teacherDifficultyChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: rows.map(item => formatGradeDisplay(item.grade, toTitleCase(item.grade))),
      datasets: [
        {
          type: 'bar',
          label: 'Average Score %',
          data: rows.map(item => item.avg_score),
          backgroundColor: ['#60a5fa', '#34d399', '#f59e0b', '#a78bfa', '#94a3b8'],
          borderColor: ['#2563eb', '#059669', '#d97706', '#7c3aed', '#64748b'],
          borderWidth: 1,
          borderRadius: 8,
          yAxisID: 'y',
        },
        {
          type: 'line',
          label: 'Active Students',
          data: rows.map(item => item.active_students),
          borderColor: '#0f172a',
          backgroundColor: 'rgba(15, 23, 42, 0.12)',
          pointRadius: 4,
          tension: 0.25,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'bottom' } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: value => `${value}%` }, title: { display: true, text: 'Average Score' } },
        y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Active Students' }, precision: 0 },
      },
    },
  });
}

function renderDifficultyInsights(items) {
  const wrap = document.getElementById('difficultyInsightsList');
  if (!wrap) return;
  const rows = Array.isArray(items) ? items : [];
  if (rows.length === 0) {
    wrap.innerHTML = '<div class="text-muted">No difficulty insight data for selected filters.</div>';
    return;
  }

  const difficultyColor = {
    easy: '#16a34a',
    medium: '#f59e0b',
    hard: '#dc2626',
    unknown: '#64748b',
  };

  wrap.innerHTML = rows.map(item => {
    const difficulty = String(item.difficulty || 'unknown').toLowerCase();
    const color = difficultyColor[difficulty] || difficultyColor.unknown;
    return `
      <div class="difficulty-insight-card" style="--difficulty-accent:${color}">
        <div class="difficulty-title">${toTitleCase(difficulty)}</div>
        <div class="small text-muted">Attempts: ${Number(item.total_attempts || 0)}</div>
        <div class="difficulty-accuracy">${formatPercent(item.accuracy || 0)}</div>
      </div>
    `;
  }).join('');
}

function renderWeakTopics(items) {
  const wrap = document.getElementById('weakTopicsList');
  if (!wrap) return;
  const rows = Array.isArray(items) ? items : [];

  if (rows.length === 0) {
    wrap.innerHTML = '<div class="text-muted">No weak topics identified in the selected period.</div>';
    return;
  }

  wrap.innerHTML = rows.map(item => `
    <div class="weak-topic-item">
      <div><strong>${escapeHtml(item.topic || 'Unknown Topic')}</strong></div>
      <div class="small text-muted">Attempts: ${Number(item.total_attempts || 0)}</div>
      <div class="small">Accuracy: <strong>${Number(item.accuracy || 0).toFixed(1)}%</strong></div>
    </div>
  `).join('');
}

function renderAtRiskStudents(items) {
  const body = document.getElementById('atRiskStudentsBody');
  if (!body) return;

  const rows = Array.isArray(items) ? items : [];
  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="3" class="text-muted">At-risk predictions not available for the selected period.</td></tr>';
    return;
  }

  const safeProb = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };

  const fmtPct = (v) => `${(safeProb(v) * 100).toFixed(1)}%`;

  rows.sort((a, b) => safeProb(b.at_risk_probability) - safeProb(a.at_risk_probability));

  body.innerHTML = rows.map(item => {
    const name = escapeHtml(item.student_name || '-');
    const prob = fmtPct(item.at_risk_probability);

    const top = item.explanation?.top_features?.[0] || null;
    const topText = top
      ? `${escapeHtml(top.feature)} (${Number(top.shap_value || 0).toFixed(3)})`
      : '-';

    return `
      <tr>
        <td>${name}</td>
        <td>${prob}</td>
        <td class="small text-muted">${topText}</td>
      </tr>
    `;
  }).join('');
}

function collectReportActionPayload() {
  const subject = document.getElementById('reportSubject')?.value?.trim() || '';
  const topic = document.getElementById('actionTopic')?.value?.trim() || '';
  const days = Number(document.getElementById('reportDays')?.value || 30);
  const dueDays = Number(document.getElementById('actionDueDays')?.value || 7);
  const maxStudents = Number(document.getElementById('actionMaxStudents')?.value || 12);
  const questionCount = Number(document.getElementById('actionQuestionCount')?.value || 8);
  const threshold = Number(document.getElementById('actionAccuracyThreshold')?.value || 60);

  return {
    subject: subject || undefined,
    topic: topic || undefined,
    days,
    due_days: dueDays,
    max_students: maxStudents,
    question_count: questionCount,
    accuracy_threshold: threshold,
    low_accuracy_threshold: threshold,
  };
}

function showReportActionResult(result, tone = 'success') {
  const el = document.getElementById('reportQuickActionResult');
  if (!el) return;

  const action = toTitleCase(String(result?.action || 'action').replace(/_/g, ' '));
  const students = Number(result?.target_student_count || 0);
  const assignments = Number(result?.created_assignments || 0);
  const clusters = Number(result?.cluster_count || 0);
  const testTitle = result?.test?.title ? String(result.test.title) : '';
  const warning = result?.warning ? String(result.warning) : '';

  const summaryParts = [action];
  if (students > 0) summaryParts.push(`Students: ${students}`);
  if (assignments > 0) summaryParts.push(`Assignments: ${assignments}`);
  if (clusters > 0) summaryParts.push(`Clusters: ${clusters}`);
  if (testTitle) summaryParts.push(`Test: ${testTitle}`);

  el.classList.remove('hidden', 'is-warning', 'is-error');
  if (tone === 'warning') {
    el.classList.add('is-warning');
  } else if (tone === 'error') {
    el.classList.add('is-error');
  }

  el.innerHTML = `
    <div class="fw-semibold">${escapeHtml(result?.message || 'Action completed')}</div>
    <div class="small mt-1">${escapeHtml(summaryParts.join(' | '))}</div>
    ${warning ? `<div class="small mt-1 text-warning-emphasis">${escapeHtml(warning)}</div>` : ''}
  `;
}

function buildInterventionStatusOptions(selected) {
  const statuses = [
    { value: 'planned', label: 'Planned' },
    { value: 'in_progress', label: 'In Progress' },
    { value: 'monitoring', label: 'Monitoring' },
    { value: 'completed', label: 'Completed' },
    { value: 'cancelled', label: 'Cancelled' },
  ];

  return statuses.map(status => {
    const isSelected = status.value === selected ? 'selected' : '';
    return `<option value="${status.value}" ${isSelected}>${status.label}</option>`;
  }).join('');
}

function renderInterventionsTable() {
  const body = document.getElementById('interventionsTableBody');
  if (!body) return;

  const rows = Array.isArray(teacherCache.interventions) ? teacherCache.interventions : [];
  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="9" class="text-muted">No intervention notes yet. Add one or run a quick action from reports.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(item => {
    const id = Number(item.id || 0);
    const status = String(item.status || 'planned').toLowerCase();
    const subjectTopic = [item.subject, item.topic].filter(Boolean).join(' / ') || '-';
    const studentCount = Array.isArray(item.student_ids) ? item.student_ids.length : 0;
    const actionTypeLabel = toTitleCase(String(item.action_type || 'note').replace(/_/g, ' '));

    return `
      <tr>
        <td>${escapeHtml(actionTypeLabel)}</td>
        <td>${escapeHtml(item.title || '-')}</td>
        <td>
          <select class="form-control form-control-sm intervention-status" data-id="${id}">
            ${buildInterventionStatusOptions(status)}
          </select>
        </td>
        <td>${escapeHtml(subjectTopic)}</td>
        <td>${studentCount}</td>
        <td><input type="datetime-local" class="form-control form-control-sm intervention-due" data-id="${id}" value="${escapeHtml(toDateTimeLocalValue(item.due_at))}" /></td>
        <td><textarea class="form-control form-control-sm intervention-notes" data-id="${id}" rows="2">${escapeHtml(item.notes || '')}</textarea></td>
        <td>${formatDate(item.updated_at || item.created_at)}</td>
        <td><button class="btn btn-sm btn-outline-primary intervention-save" data-id="${id}">Save</button></td>
      </tr>
    `;
  }).join('');
}

async function loadInterventions() {
  const result = await api.teacher.getInterventions({ limit: 200 });
  teacherCache.interventions = Array.isArray(result.items) ? result.items : [];
  renderInterventionsTable();
}

function renderAnalyticsSummary(summary) {
  const data = summary || {};
  const top = data.top_student || null;

  const totalStudents = document.getElementById('analyticsTotalStudents');
  const activeStudents = document.getElementById('analyticsActiveStudents');
  const totalAttempts = document.getElementById('analyticsTotalAttempts');
  const averageScore = document.getElementById('analyticsAverageScore');
  const topStudent = document.getElementById('analyticsTopStudent');

  if (totalStudents) totalStudents.textContent = String(Number(data.total_students || 0));
  if (activeStudents) activeStudents.textContent = String(Number(data.active_students || 0));
  if (totalAttempts) totalAttempts.textContent = String(Number(data.total_attempts || 0));
  if (averageScore) averageScore.textContent = formatPercent(data.average_score || 0);
  if (topStudent) {
    topStudent.textContent = top ? `${top.student_name} (${formatPercent(top.avg_score)})` : '-';
  }
}

function renderStudentAnalyticsTable(items) {
  const body = document.getElementById('analyticsStudentsBody');
  if (!body) return;

  const rows = Array.isArray(items) ? items : [];
  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="6" class="text-muted">No student analytics found for selected filters.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(item => `
    <tr>
      <td>${escapeHtml(item.student_name || '-')}</td>
      <td>${escapeHtml(formatGradeDisplay(item.grade))}</td>
      <td>${Number(item.total_attempts || 0)}</td>
      <td>${formatPercent(item.avg_score || 0)}</td>
      <td>${Number(item.correct_answers || 0)}/${Number(item.total_questions || 0)}</td>
      <td>${escapeHtml(item.best_subject || '-')}</td>
    </tr>
  `).join('');
}

function renderSubjectAnalyticsTable(items) {
  const body = document.getElementById('analyticsSubjectsBody');
  if (!body) return;

  const rows = Array.isArray(items) ? items : [];
  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="4" class="text-muted">No subject analytics found for selected filters.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(item => {
    const top = item.top_student || null;
    const topLabel = top ? `${top.student_name} (${formatPercent(top.avg_score)})` : '-';
    return `
      <tr>
        <td>${escapeHtml(item.subject || '-')}</td>
        <td>${Number(item.total_attempts || 0)}</td>
        <td>${formatPercent(item.avg_score || 0)}</td>
        <td>${escapeHtml(topLabel)}</td>
      </tr>
    `;
  }).join('');
}

async function loadOverview() {
  const data = await api.teacher.getDashboard(30);
  document.getElementById('statStudents').textContent = Number(data.students?.total || 0);
  document.getElementById('statTests').textContent = Number(data.tests?.total || 0);
  document.getElementById('statAttempts').textContent = Number(data.performance?.total_attempts || 0);
  document.getElementById('statAvgScore').textContent = `${Number(data.performance?.average_score || 0).toFixed(1)}%`;
}

async function loadTests() {
  const body = document.getElementById('testsTableBody');
  if (!body) return;

  const result = await api.teacher.getTests();
  const rows = Array.isArray(result.tests) ? result.tests : [];
  teacherCache.tests = rows;
  syncReportSubjectOptions();

  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="8" class="text-muted">No tests created yet.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(item => {
    const isPublished = Boolean(item.is_published);
    const statusClass = isPublished ? 'badge-published' : 'badge-draft';
    const statusText = isPublished ? 'Published' : 'Draft';
    return `
      <tr>
        <td>${escapeHtml(item.title)}</td>
        <td>${escapeHtml(item.subject)}</td>
        <td>${escapeHtml(formatGradeDisplay(item.grade))}</td>
        <td>${escapeHtml(item.difficulty)}</td>
        <td>${Number(item.attempts || 0)}</td>
        <td>${Number(item.average_score || 0).toFixed(1)}%</td>
        <td><span class="badge-soft ${statusClass}">${statusText}</span></td>
        <td class="d-flex gap-2">
          <button class="btn btn-sm btn-outline-secondary" data-action="view" data-id="${item.id}">View</button>
          <button class="btn btn-sm btn-outline-primary" data-action="toggle-publish" data-id="${item.id}" data-value="${isPublished ? '0' : '1'}">${isPublished ? 'Unpublish' : 'Publish'}</button>
          <button class="btn btn-sm btn-outline-danger" data-action="delete" data-id="${item.id}">Delete</button>
        </td>
      </tr>
    `;
  }).join('');
}

function updateCachedTestPublishState(testId, publishState) {
  const normalizedId = Number(testId);
  const nextState = Boolean(publishState);
  teacherCache.tests = (teacherCache.tests || []).map(row => {
    if (Number(row.id) !== normalizedId) return row;
    return {
      ...row,
      is_published: nextState,
    };
  });
}

function applyPublishStateToTestRow(button, publishState) {
  if (!button) return;

  const row = button.closest('tr');
  const badge = row ? row.querySelector('.badge-soft') : null;
  const isPublished = Boolean(publishState);

  if (badge) {
    badge.classList.toggle('badge-published', isPublished);
    badge.classList.toggle('badge-draft', !isPublished);
    badge.textContent = isPublished ? 'Published' : 'Draft';
  }

  button.dataset.value = isPublished ? '0' : '1';
  button.textContent = isPublished ? 'Unpublish' : 'Publish';
}

async function loadStudents() {
  const body = document.getElementById('studentsTableBody');
  if (!body) return;

  const result = await api.teacher.getStudents();
  const rows = Array.isArray(result.students) ? result.students : [];
  teacherCache.students = rows;

  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="5" class="text-muted">No students found for your scope.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(item => `
    <tr>
      <td>${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.email)}</td>
      <td>${escapeHtml(formatGradeDisplay(item.grade))}</td>
      <td>${Number(item.attempts || 0)}</td>
      <td>${Number(item.avg_score || 0).toFixed(1)}%</td>
    </tr>
  `).join('');
}

function renderClassroomsTable() {
  const body = document.getElementById('classroomsTableBody');
  if (!body) return;

  const classrooms = teacherCache.classrooms || [];
  const allStudents = getKnownStudents();

  if (classrooms.length === 0) {
    body.innerHTML = '<tr><td colspan="4" class="text-muted">No classrooms created yet.</td></tr>';
    return;
  }

  body.innerHTML = classrooms.map(item => {
    const enrolled = Array.isArray(item.students) ? item.students : [];
    const enrolledIds = new Set(enrolled.map(s => Number(s.id)));
    const candidateRows = allStudents.filter(s => !enrolledIds.has(Number(s.id)));
    const options = candidateRows.map(s => `<option value="${s.id}">${escapeHtml(s.name)} (${escapeHtml(s.email)})</option>`).join('');

    return `
      <tr>
        <td>${escapeHtml(item.name)}</td>
        <td>${escapeHtml(formatGradeDisplay(item.grade))}</td>
        <td>
          <div class="classroom-student-summary">
            <span class="classroom-student-count">${Number(item.student_count || 0)}</span>
            <button class="btn btn-sm btn-outline-secondary classroom-view-students" data-classroom-id="${item.id}">View Students</button>
          </div>
        </td>
        <td>
          <div class="d-flex gap-2">
            <select class="form-control form-control-sm classroom-student-select" data-classroom-id="${item.id}">
              <option value="">Select student</option>
              ${options}
            </select>
            <button class="btn btn-sm btn-outline-primary classroom-add-student" data-classroom-id="${item.id}">Add</button>
            <button class="btn btn-sm btn-outline-secondary classroom-enroll-grade" data-classroom-id="${item.id}">Enroll Grade</button>
          </div>
        </td>
      </tr>
    `;
  }).join('');
}

function ensureClassroomStudentsModal() {
  let modal = document.getElementById('classroomStudentsModal');
  if (modal) return modal;

  modal = document.createElement('div');
  modal.id = 'classroomStudentsModal';
  modal.className = 'classroom-students-modal hidden';
  modal.innerHTML = `
    <div class="classroom-students-backdrop" data-close="1"></div>
    <div class="classroom-students-dialog" role="dialog" aria-modal="true" aria-labelledby="classroomStudentsModalTitle">
      <div class="classroom-students-header">
        <h5 id="classroomStudentsModalTitle" class="mb-0">Classroom Students</h5>
        <button type="button" class="btn btn-sm btn-outline-secondary classroom-students-close" data-close="1">Close</button>
      </div>
      <div class="classroom-students-body" id="classroomStudentsModalBody"></div>
    </div>
  `;

  modal.addEventListener('click', (evt) => {
    const shouldClose = evt.target && evt.target.getAttribute && evt.target.getAttribute('data-close') === '1';
    if (shouldClose) {
      modal.classList.add('hidden');
      document.body.classList.remove('modal-open');
    }
  });

  document.body.appendChild(modal);
  return modal;
}

function showClassroomStudentsModal(classroomId) {
  const modal = ensureClassroomStudentsModal();
  const body = document.getElementById('classroomStudentsModalBody');
  const title = document.getElementById('classroomStudentsModalTitle');
  if (!modal || !body || !title) return;

  const classroom = (teacherCache.classrooms || []).find(c => Number(c.id) === Number(classroomId));
  const students = Array.isArray(classroom?.students) ? classroom.students : [];

  title.textContent = classroom?.name ? `${classroom.name} Students` : 'Classroom Students';

  if (students.length === 0) {
    body.innerHTML = '<div class="text-muted">No students enrolled in this classroom yet.</div>';
  } else {
    body.innerHTML = `
      <div class="table-responsive">
        <table class="table table-sm align-middle mb-0">
          <thead><tr><th>Name</th><th>Email</th><th>Grade</th><th>Action</th></tr></thead>
          <tbody>
            ${students.map(s => `
              <tr>
                <td>${escapeHtml(s.name || '-')}</td>
                <td>${escapeHtml(s.email || '-')}</td>
                <td>${escapeHtml(formatGradeDisplay(s.grade))}</td>
                <td><button class="btn btn-sm btn-outline-danger classroom-remove-student" data-classroom-id="${Number(classroomId)}" data-student-id="${Number(s.id)}">Remove</button></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  modal.classList.remove('hidden');
  document.body.classList.add('modal-open');
}

async function loadClassrooms() {
  const response = await api.teacher.getClassrooms();
  teacherCache.classrooms = Array.isArray(response.classrooms) ? response.classrooms : [];
  renderClassroomsTable();
}

function renderAssignmentsTable() {
  const body = document.getElementById('assignmentsTableBody');
  if (!body) return;

  const rows = teacherCache.assignments || [];
  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="6" class="text-muted">No assignments yet.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(item => {
    const target = item.student?.name
      ? `Student: ${item.student.name}`
      : (item.classroom?.name ? `Classroom: ${item.classroom.name}` : 'Unknown');
    const dueValue = toDateTimeLocalValue(item.due_at);
    return `
      <tr>
        <td>${escapeHtml(item.test?.title || `Test #${item.test_id}`)}</td>
        <td>${escapeHtml(target)}</td>
        <td>${escapeHtml(item.status || 'assigned')}</td>
        <td>${formatDate(item.due_at)}</td>
        <td>
          <div class="d-flex gap-2 align-items-center">
            <input type="datetime-local" class="form-control form-control-sm assignment-due-input" data-id="${item.id}" value="${escapeHtml(dueValue)}" style="min-width: 180px;" />
            <button class="btn btn-sm btn-outline-primary assignment-save-due" data-id="${item.id}">Save</button>
            <button class="btn btn-sm btn-outline-secondary assignment-clear-due" data-id="${item.id}">Clear</button>
          </div>
        </td>
        <td class="d-flex gap-2">
          <button class="btn btn-sm btn-outline-danger assignment-delete" data-id="${item.id}">Delete</button>
        </td>
      </tr>
    `;
  }).join('');
}

function syncAssignmentFormOptions() {
  const testSelect = document.getElementById('assignmentTestId');
  const classroomSelect = document.getElementById('assignmentClassroomId');
  const studentSelect = document.getElementById('assignmentStudentId');
  const dueInput = document.getElementById('assignmentDueAt');
  const mandatorySelect = document.getElementById('assignmentMandatory');
  const allowLateSelect = document.getElementById('assignmentAllowLate');
  const notesInput = document.getElementById('assignmentNotes');

  const prev = {
    test_id: testSelect?.value || '',
    classroom_id: classroomSelect?.value || '',
    student_id: studentSelect?.value || '',
    due_at: dueInput?.value || '',
    mandatory: mandatorySelect?.value,
    allow_late: allowLateSelect?.value,
    notes: notesInput?.value ?? '',
  };

  const knownStudents = getKnownStudents();

  const optionExists = (select, value) => {
    if (!select || value === '' || value === null || value === undefined) return false;
    return Array.from(select.options || []).some(opt => String(opt.value) === String(value));
  };

  if (testSelect) {
    const published = (teacherCache.tests || []).filter(t => t.is_published);
    testSelect.innerHTML = '<option value="">Select test</option>' + published.map(t => `<option value="${t.id}">${escapeHtml(t.title)} (${escapeHtml(t.subject)})</option>`).join('');
    if (optionExists(testSelect, prev.test_id)) testSelect.value = String(prev.test_id);
  }

  if (classroomSelect) {
    classroomSelect.innerHTML = '<option value="">No classroom target</option>' + (teacherCache.classrooms || []).map(c => `<option value="${c.id}">${escapeHtml(c.name)}</option>`).join('');
    if (optionExists(classroomSelect, prev.classroom_id)) classroomSelect.value = String(prev.classroom_id);
  }

  if (studentSelect) {
    studentSelect.innerHTML = '<option value="">No direct student target</option>' + knownStudents.map(s => `<option value="${s.id}">${escapeHtml(s.name)} (${escapeHtml(s.email)})</option>`).join('');
    if (optionExists(studentSelect, prev.student_id)) studentSelect.value = String(prev.student_id);
  }

  if (dueInput && prev.due_at) dueInput.value = prev.due_at;
  if (mandatorySelect && prev.mandatory !== undefined && optionExists(mandatorySelect, prev.mandatory)) {
    mandatorySelect.value = prev.mandatory;
  }
  if (allowLateSelect && prev.allow_late !== undefined && optionExists(allowLateSelect, prev.allow_late)) {
    allowLateSelect.value = prev.allow_late;
  }
  if (notesInput && prev.notes !== undefined) notesInput.value = prev.notes;
}

async function loadAssignments() {
  const response = await api.teacher.getAssignments();
  teacherCache.assignments = Array.isArray(response.assignments) ? response.assignments : [];
  renderAssignmentsTable();
}

async function loadReports() {
  const body = document.getElementById('teacherReportsBody');
  if (!body) return;

  const subject = document.getElementById('reportSubject')?.value?.trim() || '';
  const days = Number(document.getElementById('reportDays')?.value || 30);
  const limit = Number(document.getElementById('reportLimit')?.value || 100);

  const result = await api.teacher.getReports({ subject, days, limit });
  const rows = Array.isArray(result.items) ? result.items : [];
  teacherCache.reportRows = rows;
  syncReportSubjectOptions();

  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="6" class="text-muted">No report rows for selected filters.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(item => `
    <tr>
      <td>${escapeHtml(item.student_name)}</td>
      <td>${escapeHtml(item.subject)}</td>
      <td>${Number(item.score_pct || 0).toFixed(1)}%</td>
      <td>${Number(item.correct_answers || 0)}/${Number(item.total_questions || 0)}</td>
      <td>${escapeHtml(item.status || '-')}</td>
      <td>${formatDate(item.test_date)}</td>
    </tr>
  `).join('');
}

async function loadAnalytics() {
  const days = Number(document.getElementById('analyticsDays')?.value || document.getElementById('reportDays')?.value || 30);
  const grade = document.getElementById('analyticsGrade')?.value || '';
  const result = await api.teacher.getAnalytics({ days, grade });
  const gradeRows = Array.isArray(result.grade_performance) && result.grade_performance.length > 0
    ? result.grade_performance
    : buildGradePerformanceFromStudents(result.student_performance || []);

  renderAnalyticsSummary(result.summary || {});
  renderStudentAnalyticsTable(result.student_performance || []);
  renderSubjectAnalyticsTable(result.subject_performance || []);
  renderSubjectChart(result.subject_performance || []);
  renderGradeLevelChart(gradeRows);
  renderDifficultyInsights(result.difficulty_performance || []);
  renderWeakTopics(result.weak_topics || []);
  renderAtRiskStudents(result.at_risk_students || []);
}

function buildQuestionCitationMarkup(question) {
  const provenance = question && typeof question.provenance === 'object' ? question.provenance : null;
  const trace = Array.isArray(question?.retrieval_trace) ? question.retrieval_trace : [];

  if (!provenance && trace.length === 0) {
    return `
      <div class="question-citations">
        <div class="question-citations-title">Source Citations</div>
        <div class="citation-empty">No retrieval citation was attached to this question.</div>
      </div>
    `;
  }

  const topLine = provenance
    ? `<div class="citation-topline"><strong>${escapeHtml(provenance.document_title || 'Document')}</strong> • Chunk ${(Number(provenance.chunk_index || 0) + 1)} • Similarity ${Number(provenance.similarity || 0).toFixed(3)}</div>`
    : '';

  const traceItems = trace.slice(0, 3).map(item => {
    const similarity = Number(item?.similarity || 0).toFixed(3);
    const docTitle = escapeHtml(item?.document_title || 'Document');
    const chunk = Number(item?.chunk_index || 0) + 1;
    const snippet = escapeHtml(item?.snippet || '');
    return `
      <div class="citation-trace-item">
        <div class="citation-trace-meta">
          <span>${docTitle} • Chunk ${chunk}</span>
          <strong>${similarity}</strong>
        </div>
        <div class="citation-trace-snippet">${snippet || 'No snippet available.'}</div>
      </div>
    `;
  }).join('');

  return `
    <div class="question-citations">
      <div class="question-citations-title">Source Citations</div>
      ${topLine}
      <div class="citation-trace-list">${traceItems}</div>
    </div>
  `;
}

function bindTestActions() {
  const body = document.getElementById('testsTableBody');
  if (!body) return;

  body.addEventListener('click', async evt => {
    const btn = evt.target.closest('button[data-action]');
    if (!btn) return;

    const action = btn.dataset.action;
    const testId = Number(btn.dataset.id);
    if (!testId) return;

    try {
      if (action === 'view') {
        const panel = document.getElementById('testDetailPanel');
        if (!panel) return;

        // Toggle Logic: If clicking View on an already open test, hide it.
        if (panel.dataset.currentTestId === String(testId) && !panel.classList.contains('hidden')) {
          panel.classList.add('hidden');
          btn.textContent = 'View';
          return;
        }

        const detail = await api.teacher.getTest(testId);
        const questions = Array.isArray(detail.questions) ? detail.questions : [];
        
        panel.dataset.currentTestId = testId;
        panel.classList.remove('hidden');
        
        // Change button states
        document.querySelectorAll('button[data-action="view"]').forEach(b => {
          b.textContent = b.dataset.id === String(testId) ? 'Hide' : 'View';
        });

        panel.innerHTML = `
          <div class="d-flex justify-content-between align-items-center mb-2">
            <h5 class="mb-0">${escapeHtml(detail.title)} Questions</h5>
            <button class="btn btn-sm btn-primary" id="saveTestQuestionsBtn" data-test-id="${testId}">Update Questions</button>
          </div>
          <div class="small text-muted mb-2">Drag by the serial card to reorder. Edit question, points, answers, explanation, and correct answer. Then click Update Questions.</div>
          <div class="test-question-editor-list d-flex flex-column gap-3">
            ${questions.map((q, idx) => {
              const opts = Array.isArray(q.options) ? q.options : [];
              const safeOpts = [0, 1, 2, 3].map(i => escapeHtml(opts[i] || ''));
              const orderVal = Number(q.order || (idx + 1));
              const pointsVal = Number(q.points || 1);
              const correctVal = Number(q.correct_index || 0);
              const clampedCorrect = Math.max(0, Math.min(3, correctVal));
              const citationMarkup = buildQuestionCitationMarkup(q);
              const optionsMarkup = safeOpts.map((optText, optIndex) => `
                    <div class="option-row q-option-row" data-option-index="${optIndex}" draggable="true">
                      <button type="button" class="option-drag-handle" title="Drag to reorder option" aria-label="Drag option">≡</button>
                      <span class="option-label">${OPTION_LETTERS[optIndex]}</span>
                      <input type="text" class="form-control form-control-sm q-option-input" placeholder="Option ${optIndex + 1}" value="${optText}" />
                    </div>
              `).join('');
              
              return `
                <article class="test-question-editor-row" data-question-id="${q.id}">
                  <div class="question-card-head">
                    <button type="button" class="question-serial-card" title="Drag to reorder">
                      <span class="question-serial-prefix">Q</span><span class="question-serial-number">${orderVal}</span>
                    </button>
                    <div class="question-points-wrap">
                      <label class="form-label small mb-1">Points</label>
                      <input type="number" class="form-control form-control-sm q-points" min="1" value="${pointsVal}" />
                    </div>
                  </div>
                  <input type="hidden" class="q-order" value="${orderVal}" />
                  <div class="question-main-field">
                    <label class="form-label small mb-1">Question</label>
                    <textarea class="form-control q-text" rows="2">${escapeHtml(q.text || '')}</textarea>
                  </div>
                  <div class="question-options-list">
                    <label class="form-label small mb-1">Answers</label>
                    ${optionsMarkup}
                  </div>
                  <div class="row mt-2">
                    <div class="col-md-3">
                      <label class="form-label small mb-1">Correct Answer</label>
                      <select class="form-control form-control-sm q-correct">
                        <option value="${clampedCorrect}" selected>Loading options...</option>
                      </select>
                    </div>
                    <div class="col-md-9">
                      <label class="form-label small mb-1">Explanation</label>
                      <textarea class="form-control form-control-sm q-explanation" rows="2" placeholder="Explain why the answer is correct...">${escapeHtml(q.explanation || '')}</textarea>
                    </div>
                  </div>
                  ${citationMarkup}
                </article>
              `;
            }).join('')}
          </div>
        `;

        enableQuestionCardDragAndDrop(panel);
        initializeOptionEditors(panel);

        const saveBtn = panel.querySelector('#saveTestQuestionsBtn');
        if (saveBtn) {
          saveBtn.addEventListener('click', async () => {
            try {
              const questionPayload = collectEditableQuestionsFromPanel(panel);
              if (questionPayload.length === 0) {
                utils.showNotification('No questions to update.', 'warning');
                return;
              }

              await api.teacher.updateTest(testId, { questions: questionPayload });
              notifyTeacherSuccess('testUpdateSuccess');
              
              // Persist the "Hide" button state after reloading
              await loadTests();
              document.querySelectorAll(`button[data-action="view"][data-id="${testId}"]`).forEach(b => b.textContent = 'Hide');
            } catch (error) {
              console.error(error);
              notifyTeacherError('testUpdateFailed', error);
            }
          });
        }
      }

      if (action === 'toggle-publish') {
        const publish = btn.dataset.value === '1';
        const panel = document.getElementById('testDetailPanel');
        const currentOpenTestId = panel && !panel.classList.contains('hidden')
          ? Number(panel.dataset.currentTestId || 0)
          : 0;

        btn.disabled = true;
        btn.textContent = publish ? 'Publishing...' : 'Unpublishing...';

        await api.teacher.updateTest(testId, { is_published: publish });
        updateCachedTestPublishState(testId, publish);
        applyPublishStateToTestRow(btn, publish);
        syncAssignmentFormOptions();
        notifyTeacherSuccess('testUpdateSuccess');

        loadTests()
          .then(() => {
            syncAssignmentFormOptions();
            if (currentOpenTestId > 0) {
              document.querySelectorAll(`button[data-action="view"][data-id="${currentOpenTestId}"]`).forEach(b => {
                b.textContent = 'Hide';
              });
            }
          })
          .catch(error => {
            console.warn('Failed to revalidate tests after publish toggle:', error);
          });

        refreshAssessmentsRealtime({ force: true }).catch(() => {});

        if (btn.isConnected) {
          btn.disabled = false;
        }
      }

      if (action === 'delete') {
        if (!window.confirm('Delete this test? This action cannot be undone.')) return;
        await api.teacher.deleteTest(testId);
        notifyTeacherSuccess('testDeleteSuccess');

        // Hide panel if the deleted test was being viewed
        const panel = document.getElementById('testDetailPanel');
        if (panel && panel.dataset.currentTestId === String(testId)) {
          panel.innerHTML = '';
          panel.classList.add('hidden');
          panel.dataset.currentTestId = '';
        }

        await loadTests();
        syncAssignmentFormOptions();
        refreshAssessmentsRealtime({ force: true }).catch(() => {});
      }
    } catch (error) {
      if (action === 'toggle-publish' && btn && btn.isConnected) {
        btn.disabled = false;
        btn.textContent = btn.dataset.value === '1' ? 'Publish' : 'Unpublish';
      }
      console.error(error);
      notifyTeacherError('testUpdateFailed', error);
    }
  });
}

function formatFileSize(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function getSelectedRagDocumentIds() {
  return Array.from(document.querySelectorAll('.rag-doc-select:checked'))
    .map(input => Number(input.value))
    .filter(id => Number.isFinite(id) && id > 0);
}

function getAssessmentMode() {
  return activeAssessmentMode === 'document' ? 'document' : 'topic';
}

function setAssessmentMode(mode) {
  const normalized = String(mode || '').toLowerCase() === 'document' ? 'document' : 'topic';
  activeAssessmentMode = normalized;

  document.querySelectorAll('.assessment-mode-btn').forEach(btn => {
    btn.classList.toggle('is-active', btn.dataset.mode === normalized);
  });

  const topicPanel = document.getElementById('topicBuilderPanel');
  const documentPanel = document.getElementById('documentBuilderPanel');
  if (topicPanel) topicPanel.classList.toggle('hidden', normalized !== 'topic');
  if (documentPanel) documentPanel.classList.toggle('hidden', normalized !== 'document');

  syncRagModeControls();
}

function syncRagModeControls() {
  const mode = getAssessmentMode();
  const ragTopK = document.getElementById('ragTopK');
  const summary = document.getElementById('ragDocSummary');
  const isRag = mode === 'document';

  if (ragTopK) {
    ragTopK.disabled = !isRag;
  }

  document.querySelectorAll('.rag-doc-select').forEach(input => {
    input.disabled = !isRag || input.dataset.status !== 'processed';
  });

  if (summary) {
    const docs = teacherCache.documents || [];
    const selectedCount = getSelectedRagDocumentIds().length;
    const processedCount = docs.filter(item => String(item.status || '').toLowerCase() === 'processed').length;

    if (!isRag) {
      summary.textContent = `${docs.length} documents indexed. Switch to Document Based Generation to use them.`;
    } else if (selectedCount > 0) {
      summary.textContent = `${selectedCount} selected of ${processedCount} processed document(s).`;
    } else {
      summary.textContent = processedCount > 0
        ? `No documents selected. RAG will use all ${processedCount} processed document(s).`
        : 'No processed documents available yet. Upload a file and wait until indexing completes.';
    }
  }
}

function renderRagDocuments() {
  const container = document.getElementById('ragDocList');
  if (!container) return;

  const docs = Array.isArray(teacherCache.documents) ? teacherCache.documents : [];
  if (docs.length === 0) {
    container.innerHTML = '<div class="rag-doc-empty">No documents uploaded yet. Upload .pdf, .doc, .docx, or .txt resources.</div>';
    syncRagModeControls();
    return;
  }

  container.innerHTML = docs.map(doc => {
    const status = String(doc.status || 'processed').toLowerCase();
    const statusClass = status === 'failed' ? 'is-failed' : (status === 'processing' ? 'is-processing' : 'is-processed');
    const checkboxDisabled = status !== 'processed' ? 'disabled' : '';
    const title = doc.title || doc.filename || `Document #${doc.id}`;
    const meta = [
      `${Number(doc.chunk_count || 0)} chunks`,
      formatFileSize(doc.file_size_bytes),
      formatDate(doc.uploaded_at),
    ].join(' • ');

    return `
      <div class="rag-doc-item">
        <div class="rag-doc-main">
          <input class="form-check-input rag-doc-select rag-doc-checkbox" type="checkbox" value="${Number(doc.id)}" data-status="${escapeHtml(status)}" ${checkboxDisabled} />
          <div class="rag-doc-info">
            <div class="rag-doc-title">${escapeHtml(title)}</div>
            <div class="rag-doc-meta">${escapeHtml(meta)}</div>
            ${doc.error_message ? `<div class="rag-doc-meta text-danger">${escapeHtml(doc.error_message)}</div>` : ''}
          </div>
        </div>
        <div class="rag-doc-actions">
          <span class="rag-doc-status ${statusClass}">${escapeHtml(status)}</span>
          <button class="btn btn-sm btn-outline-danger rag-doc-delete" type="button" data-id="${Number(doc.id)}">Delete</button>
        </div>
      </div>
    `;
  }).join('');

  syncRagModeControls();
}

function renderRagObservability(payload) {
  teacherCache.ragObservability = payload || null;

  const totalEl = document.getElementById('ragObsTotalEvents');
  const fallbackEl = document.getElementById('ragObsFallbackRate');
  const confidenceEl = document.getElementById('ragObsConfidence');
  const latencyEl = document.getElementById('ragObsLatency');
  const summaryEl = document.getElementById('ragObservabilitySummary');
  const reasonsEl = document.getElementById('ragFallbackReasonList');

  const summary = payload && typeof payload.summary === 'object' ? payload.summary : {};
  const documents = payload && typeof payload.documents === 'object' ? payload.documents : {};
  const reasonRows = Array.isArray(payload?.fallback_reasons) ? payload.fallback_reasons : [];

  const totalEvents = Number(summary.total_events || 0);
  const fallbackRate = Number(summary.fallback_rate || 0);
  const avgConfidence = Number(summary.avg_confidence || 0);
  const avgLatency = Number(summary.avg_latency_ms || 0);

  if (totalEl) totalEl.textContent = String(totalEvents);
  if (fallbackEl) fallbackEl.textContent = `${(fallbackRate * 100).toFixed(1)}%`;
  if (confidenceEl) confidenceEl.textContent = avgConfidence.toFixed(2);
  if (latencyEl) latencyEl.textContent = `${avgLatency.toFixed(0)}ms`;

  if (summaryEl) {
    const statusCounts = documents.status_counts || {};
    const processed = Number(statusCounts.processed || 0);
    const processing = Number(statusCounts.processing || 0);
    const failed = Number(statusCounts.failed || 0);
    summaryEl.textContent = (
      `Events: ${totalEvents} • Success: ${Number(summary.success_events || 0)} • ` +
      `Fallback: ${Number(summary.fallback_events || 0)} • Errors: ${Number(summary.error_events || 0)} • ` +
      `Docs: ${processed} processed, ${processing} processing, ${failed} failed`
    );
  }

  if (reasonsEl) {
    if (reasonRows.length === 0) {
      reasonsEl.innerHTML = '';
    } else {
      reasonsEl.innerHTML = reasonRows.slice(0, 6).map(item => {
        const reason = escapeHtml(item?.reason || 'fallback');
        const count = Number(item?.count || 0);
        return `<span class="rag-fallback-badge">${reason} (${count})</span>`;
      }).join('');
    }
  }
}

async function loadRagObservability() {
  const days = Number(document.getElementById('reportDays')?.value || document.getElementById('analyticsDays')?.value || 30);
  const result = await api.teacher.getRagObservability({ days, limit: 500 });
  renderRagObservability(result || {});
}

async function loadTeacherDocuments() {
  const result = await api.teacher.getDocuments();
  teacherCache.documents = Array.isArray(result.documents) ? result.documents : [];
  renderRagDocuments();
}

async function uploadTeacherDocument() {
  const fileInput = document.getElementById('ragUploadFile');
  const titleInput = document.getElementById('ragUploadTitle');
  const button = document.getElementById('ragUploadBtn');
  if (!fileInput || !button) return;

  const file = fileInput.files?.[0];
  if (!file) {
    utils.showNotification('Select a document file before uploading.', 'warning');
    return;
  }

  setBusy(button, 'Uploading');
  try {
    const title = titleInput?.value?.trim() || '';
    const response = await api.teacher.uploadDocument(file, { title });
    if (response?.queued) {
      utils.showNotification('Document uploaded. Indexing is running in the background.', 'info');
    } else if (response?.deduplicated) {
      utils.showNotification('This document was already indexed.', 'info');
    } else {
      utils.showNotification('Document uploaded and indexed.', 'success');
    }
    fileInput.value = '';
    if (titleInput) titleInput.value = '';
    await Promise.all([loadTeacherDocuments(), loadRagObservability()]);
  } catch (error) {
    console.error(error);
    notifyTeacherError('documentUploadFailed', error);
  } finally {
    clearBusy(button);
  }
}

async function deleteTeacherDocument(documentId) {
  const id = Number(documentId);
  if (!id) return;
  if (!window.confirm('Delete this document and its indexed chunks?')) return;

  try {
    await api.teacher.deleteDocument(id);
    utils.showNotification('Document deleted.', 'success');
    await Promise.all([loadTeacherDocuments(), loadRagObservability()]);
  } catch (error) {
    console.error(error);
    notifyTeacherError('documentDeleteFailed', error);
  }
}

function collectTopicBuilderPayload() {
  return {
    title: document.getElementById('testTitle')?.value?.trim(),
    description: document.getElementById('testDescription')?.value?.trim(),
    subject: document.getElementById('testSubject')?.value?.trim(),
    grade: document.getElementById('testGrade')?.value,
    difficulty: document.getElementById('testDifficulty')?.value,
    question_count: Number(document.getElementById('testCount')?.value || 10),
    time_limit: Number(document.getElementById('testTimeLimit')?.value || 30),
    topic: document.getElementById('testTopic')?.value?.trim(),
    generation_mode: 'standard',
  };
}

function collectDocumentBuilderPayload() {
  const ragTopK = Number(document.getElementById('ragTopK')?.value || 4);
  const selectedDocumentIds = getSelectedRagDocumentIds();

  const payload = {
    title: document.getElementById('ragTestTitle')?.value?.trim(),
    description: document.getElementById('ragTestDescription')?.value?.trim(),
    subject: document.getElementById('ragTestSubject')?.value?.trim(),
    grade: document.getElementById('ragTestGrade')?.value,
    difficulty: document.getElementById('ragTestDifficulty')?.value,
    question_count: Number(document.getElementById('ragTestCount')?.value || 10),
    time_limit: Number(document.getElementById('ragTestTimeLimit')?.value || 30),
    generation_mode: 'rag',
    rag_min_confidence: 0.0,
  };

  if (Number.isFinite(ragTopK)) {
    payload.rag_top_k = Math.max(1, Math.min(12, Math.floor(ragTopK)));
  }
  if (selectedDocumentIds.length > 0) {
    payload.selected_document_ids = selectedDocumentIds;
  }

  return payload;
}

async function submitTestPayload(payload) {
  const result = await api.teacher.createTest(payload);
  const diagnostics = formatGenerationDiagnostics(result?.generation_status);

  if (result && result.warning) {
    utils.showNotification(result.warning, 'warning');
  } else {
    notifyTeacherSuccess('testCreateSuccess');
  }

  if (diagnostics.summary || diagnostics.error) {
    console.info('Generation diagnostics:', diagnostics);
  }

  await Promise.all([loadOverview(), loadTests(), loadAssignments(), loadRagObservability()]);
  syncAssignmentFormOptions();
  refreshAssessmentsRealtime({ force: true }).catch(() => {});
}

async function createTopicTest(event) {
  event.preventDefault();
  const button = document.getElementById('createTopicTestBtn');
  const stopBusyTimer = setBusyWithElapsed(button, 'Creating');
  try {
    const payload = collectTopicBuilderPayload();
    if (!payload.title || !payload.subject) {
      utils.showNotification('Title and subject are required.', 'warning');
      return;
    }
    if (!payload.topic) {
      utils.showNotification('Sub topic is required for topic based generation.', 'warning');
      return;
    }

    await submitTestPayload(payload);
  } catch (error) {
    console.error(error);
    notifyTeacherError('testCreateFailed', error);
  } finally {
    stopBusyTimer();
    clearBusy(button);
  }
}

async function createDocumentTest(event) {
  event.preventDefault();
  const button = document.getElementById('createDocumentTestBtn');
  const stopBusyTimer = setBusyWithElapsed(button, 'Creating');
  try {
    const payload = collectDocumentBuilderPayload();
    if (!payload.title || !payload.subject) {
      utils.showNotification('Title and subject are required.', 'warning');
      return;
    }

    const processedCount = (teacherCache.documents || []).filter(
      item => String(item.status || '').toLowerCase() === 'processed'
    ).length;
    if (processedCount <= 0) {
      utils.showNotification('Upload and process at least one document before creating a document based test.', 'warning');
      return;
    }

    await submitTestPayload(payload);
  } catch (error) {
    console.error(error);
    notifyTeacherError('testCreateFailed', error);
  } finally {
    stopBusyTimer();
    clearBusy(button);
  }
}

async function loadAllTeacherData() {
  const results = await Promise.allSettled([
    loadOverview(),
    loadTests(),
    loadStudents(),
    loadClassrooms(),
    loadAssignments(),
    loadReports(),
    loadAnalytics(),
    loadInterventions(),
    loadTeacherDocuments(),
    loadRagObservability(),
  ]);

  const failures = results.filter(item => item.status === 'rejected');
  if (failures.length > 0) {
    console.warn('Teacher dashboard partial load failures:', failures);
    utils.showNotification('Some sections could not be loaded. You can retry from each section.', 'warning');
  }

  syncAssignmentFormOptions();
  syncRagModeControls();
}

function bindEvents() {
  const createTopicForm = document.getElementById('createTopicTestForm');
  if (createTopicForm) {
    createTopicForm.addEventListener('submit', createTopicTest);
  }

  const createDocumentForm = document.getElementById('createDocumentTestForm');
  if (createDocumentForm) {
    createDocumentForm.addEventListener('submit', createDocumentTest);
  }

  document.querySelectorAll('.assessment-mode-btn').forEach(button => {
    button.addEventListener('click', () => {
      setAssessmentMode(button.dataset.mode);
    });
  });

  const modeSwitch = document.getElementById('assessmentModeSwitch');
  if (modeSwitch) {
    modeSwitch.addEventListener('keydown', evt => {
      if (evt.key !== 'ArrowLeft' && evt.key !== 'ArrowRight') return;

      const buttons = Array.from(modeSwitch.querySelectorAll('.assessment-mode-btn'));
      const activeIndex = buttons.findIndex(btn => btn.classList.contains('is-active'));
      if (activeIndex < 0) return;

      const nextIndex = evt.key === 'ArrowRight'
        ? Math.min(buttons.length - 1, activeIndex + 1)
        : Math.max(0, activeIndex - 1);

      const next = buttons[nextIndex];
      if (!next) return;
      next.focus();
      setAssessmentMode(next.dataset.mode);
    });
  }

  const uploadDocBtn = document.getElementById('ragUploadBtn');
  if (uploadDocBtn) {
    uploadDocBtn.addEventListener('click', async () => {
      await uploadTeacherDocument();
    });
  }

  const refreshDocsBtn = document.getElementById('ragRefreshDocsBtn');
  if (refreshDocsBtn) {
    refreshDocsBtn.addEventListener('click', async () => {
      await Promise.all([loadTeacherDocuments(), loadRagObservability()]);
    });
  }

  const refreshRagObsBtn = document.getElementById('refreshRagObservabilityBtn');
  if (refreshRagObsBtn) {
    refreshRagObsBtn.addEventListener('click', async () => {
      await loadRagObservability();
    });
  }

  const ragDocList = document.getElementById('ragDocList');
  if (ragDocList) {
    ragDocList.addEventListener('change', evt => {
      if (evt.target?.classList?.contains('rag-doc-select')) {
        syncRagModeControls();
      }
    });

    ragDocList.addEventListener('click', async evt => {
      const deleteBtn = evt.target.closest('.rag-doc-delete');
      if (!deleteBtn) return;
      await deleteTeacherDocument(deleteBtn.dataset.id);
    });
  }

  const refreshReportsBtn = document.getElementById('refreshReportsBtn');
  if (refreshReportsBtn) {
    refreshReportsBtn.addEventListener('click', async () => {
      await Promise.all([loadReports(), loadAnalytics(), loadInterventions(), loadRagObservability()]);
    });
  }

  const refreshAnalyticsBtn = document.getElementById('refreshAnalyticsBtn');
  if (refreshAnalyticsBtn) {
    refreshAnalyticsBtn.addEventListener('click', async () => {
      await Promise.all([loadAnalytics(), loadRagObservability()]);
    });
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) return;
    if (activeWorkspaceTarget !== 'assessmentsWorkspace') return;
    refreshAssessmentsRealtime({ force: true }).catch(() => {});
  });

  window.addEventListener('beforeunload', () => {
    stopAssessmentsRealtimeRefresh();
  });

  const actionHandlers = [
    {
      id: 'actionRemedialBtn',
      busy: 'Assigning',
      actionLabel: 'assign_remedial_test',
      run: (payload) => api.teacher.assignRemedialTest(payload),
    },
    {
      id: 'actionFocusedPracticeBtn',
      busy: 'Creating',
      actionLabel: 'create_focused_practice_set',
      run: (payload) => api.teacher.createFocusedPractice(payload),
    },
    {
      id: 'actionFollowUpBtn',
      busy: 'Scheduling',
      actionLabel: 'schedule_follow_up_assignment',
      run: (payload) => api.teacher.scheduleFollowUpAssignment(payload),
    },
    {
      id: 'actionWeaknessClustersBtn',
      busy: 'Grouping',
      actionLabel: 'group_weakness_clusters',
      run: (payload) => api.teacher.groupWeaknessClusters(payload),
    },
  ];

  actionHandlers.forEach(cfg => {
    const button = document.getElementById(cfg.id);
    if (!button) return;

    button.addEventListener('click', async () => {
      setBusy(button, cfg.busy);
      try {
        const payload = collectReportActionPayload();
        const result = await cfg.run(payload);
        if (!result.action) {
          result.action = cfg.actionLabel;
        }

        const tone = result.warning ? 'warning' : 'success';
        showReportActionResult(result, tone);
        utils.showNotification(result.message || 'Action completed successfully.', tone === 'warning' ? 'warning' : 'success');

        await Promise.all([
          loadAssignments(),
          loadReports(),
          loadAnalytics(),
          loadInterventions(),
        ]);
      } catch (error) {
        console.error(error);
        showReportActionResult({
          message: error?.message || 'Failed to run action',
          action: cfg.actionLabel,
        }, 'error');
        utils.showNotification(error.message || 'Failed to run quick action.', 'error');
      } finally {
        clearBusy(button);
      }
    });
  });

  const interventionForm = document.getElementById('interventionForm');
  if (interventionForm) {
    interventionForm.addEventListener('submit', async evt => {
      evt.preventDefault();
      const button = document.getElementById('createInterventionBtn');
      setBusy(button, 'Saving');

      try {
        const title = document.getElementById('interventionTitle')?.value?.trim() || '';
        const status = document.getElementById('interventionStatus')?.value || 'planned';
        const subject = document.getElementById('interventionSubject')?.value?.trim() || null;
        const topic = document.getElementById('interventionTopic')?.value?.trim() || null;
        const notes = document.getElementById('interventionNotes')?.value?.trim() || null;
        const dueRaw = document.getElementById('interventionDueAt')?.value?.trim() || '';
        const dueAt = dueRaw ? new Date(dueRaw).toISOString() : null;

        if (!title) {
          utils.showNotification('Intervention title is required.', 'warning');
          return;
        }

        await api.teacher.createIntervention({
          action_type: 'note',
          title,
          status,
          subject,
          topic,
          notes,
          due_at: dueAt,
        });

        utils.showNotification('Intervention note added.', 'success');
        interventionForm.reset();
        const statusInput = document.getElementById('interventionStatus');
        if (statusInput) statusInput.value = 'planned';

        await loadInterventions();
      } catch (error) {
        console.error(error);
        utils.showNotification(error.message || 'Failed to add intervention note.', 'error');
      } finally {
        clearBusy(button);
      }
    });
  }

  const interventionsBody = document.getElementById('interventionsTableBody');
  if (interventionsBody) {
    interventionsBody.addEventListener('click', async evt => {
      const saveBtn = evt.target.closest('.intervention-save');
      if (!saveBtn) return;

      const interventionId = Number(saveBtn.dataset.id || 0);
      if (!interventionId) return;

      const statusInput = interventionsBody.querySelector(`.intervention-status[data-id="${interventionId}"]`);
      const dueInput = interventionsBody.querySelector(`.intervention-due[data-id="${interventionId}"]`);
      const notesInput = interventionsBody.querySelector(`.intervention-notes[data-id="${interventionId}"]`);

      const payload = {
        status: statusInput?.value || 'planned',
        notes: notesInput?.value?.trim() || null,
        due_at: dueInput?.value ? new Date(dueInput.value).toISOString() : null,
      };

      setBusy(saveBtn, 'Saving');
      try {
        await api.teacher.updateIntervention(interventionId, payload);
        utils.showNotification('Intervention updated.', 'success');
        await loadInterventions();
      } catch (error) {
        console.error(error);
        utils.showNotification(error.message || 'Failed to update intervention.', 'error');
      } finally {
        clearBusy(saveBtn);
      }
    });
  }

  const classroomForm = document.getElementById('classroomForm');
  if (classroomForm) {
    classroomForm.addEventListener('submit', async evt => {
      evt.preventDefault();
      const btn = document.getElementById('createClassroomBtn');
      setBusy(btn, 'Creating');
      try {
        const payload = {
          name: document.getElementById('classroomName')?.value?.trim(),
          grade: document.getElementById('classroomGrade')?.value,
          auto_enroll_students: Boolean(document.getElementById('autoEnrollGradeStudents')?.checked),
        };
        if (!payload.name) {
          utils.showNotification('Classroom name is required.', 'warning');
          return;
        }
        const result = await api.teacher.createClassroom(payload);
        if (result && result.auto_enroll_students) {
          const enrolledCount = Number(result.enrolled_count || 0);
          utils.showNotification(`Classroom created. ${enrolledCount} student(s) enrolled automatically.`, 'success');
        } else {
          notifyTeacherSuccess('classroomCreateSuccess');
        }
        classroomForm.reset();
        const autoEnroll = document.getElementById('autoEnrollGradeStudents');
        if (autoEnroll) autoEnroll.checked = true;
        await Promise.all([loadClassrooms(), loadStudents()]);
        syncAssignmentFormOptions();
      } catch (error) {
        console.error(error);
        notifyTeacherError('classroomCreateFailed', error);
      } finally {
        clearBusy(btn);
      }
    });
  }

  const classroomsBody = document.getElementById('classroomsTableBody');
  if (classroomsBody) {
    classroomsBody.addEventListener('click', async evt => {
      const viewBtn = evt.target.closest('.classroom-view-students');
      if (viewBtn) {
        const classroomId = Number(viewBtn.dataset.classroomId);
        if (!classroomId) return;
        showClassroomStudentsModal(classroomId);
        return;
      }

      const enrollGradeBtn = evt.target.closest('.classroom-enroll-grade');
      if (enrollGradeBtn) {
        const classroomId = Number(enrollGradeBtn.dataset.classroomId);
        if (!classroomId) return;

        try {
          const result = await api.teacher.enrollClassroomByGrade(classroomId);
          utils.showNotification(`Enrolled ${Number(result.enrolled_count || 0)} student(s) by grade.`, 'success');
          await Promise.all([loadClassrooms(), loadStudents()]);
          syncAssignmentFormOptions();
        } catch (error) {
          console.error(error);
          utils.showNotification(error.message || 'Failed to enroll students by grade', 'error');
        }
        return;
      }

      const removeBtn = evt.target.closest('.classroom-remove-student');
      if (removeBtn) {
        const classroomId = Number(removeBtn.dataset.classroomId);
        const studentId = Number(removeBtn.dataset.studentId);
        if (!classroomId || !studentId) return;

        try {
          await api.teacher.removeStudentFromClassroom(classroomId, studentId);
          notifyTeacherSuccess('studentRemoveSuccess');
          await Promise.all([loadClassrooms(), loadStudents()]);
          syncAssignmentFormOptions();
        } catch (error) {
          console.error(error);
          notifyTeacherError('studentRemoveFailed', error);
        }
        return;
      }

      const addBtn = evt.target.closest('.classroom-add-student');
      if (!addBtn) return;

      const classroomId = Number(addBtn.dataset.classroomId);
      const select = classroomsBody.querySelector(`.classroom-student-select[data-classroom-id="${classroomId}"]`);
      const studentId = Number(select?.value || 0);
      if (!classroomId || !studentId) {
        utils.showNotification('Select a student first.', 'warning');
        return;
      }

      try {
        await api.teacher.addStudentToClassroom(classroomId, studentId);
        notifyTeacherSuccess('studentAddSuccess');
        await Promise.all([loadClassrooms(), loadStudents()]);
        syncAssignmentFormOptions();
      } catch (error) {
        console.error(error);
        notifyTeacherError('studentAddFailed', error);
      }
    });
  }

  const modal = ensureClassroomStudentsModal();
  if (modal) {
    modal.addEventListener('click', async evt => {
      const removeBtn = evt.target.closest('.classroom-remove-student');
      if (!removeBtn) return;

      const classroomId = Number(removeBtn.dataset.classroomId);
      const studentId = Number(removeBtn.dataset.studentId);
      if (!classroomId || !studentId) return;

      try {
        await api.teacher.removeStudentFromClassroom(classroomId, studentId);
        notifyTeacherSuccess('studentRemoveSuccess');
        await Promise.all([loadClassrooms(), loadStudents()]);
        syncAssignmentFormOptions();
        showClassroomStudentsModal(classroomId);
      } catch (error) {
        console.error(error);
        notifyTeacherError('studentRemoveFailed', error);
      }
    });
  }

  const assignmentForm = document.getElementById('assignmentForm');
  if (assignmentForm) {
    assignmentForm.addEventListener('submit', async evt => {
      evt.preventDefault();
      const btn = document.getElementById('createAssignmentBtn');
      setBusy(btn, 'Assigning');
      try {
        const payload = {
          test_id: Number(document.getElementById('assignmentTestId')?.value || 0),
          classroom_id: Number(document.getElementById('assignmentClassroomId')?.value || 0) || null,
          student_id: Number(document.getElementById('assignmentStudentId')?.value || 0) || null,
          due_at: document.getElementById('assignmentDueAt')?.value ? new Date(document.getElementById('assignmentDueAt').value).toISOString() : null,
          notes: document.getElementById('assignmentNotes')?.value?.trim() || null,
          is_mandatory: document.getElementById('assignmentMandatory')?.value === '1',
          allow_late: document.getElementById('assignmentAllowLate')?.value === '1',
        };

        if (!payload.test_id) {
          utils.showNotification('Select a test to assign.', 'warning');
          return;
        }
        if (!payload.classroom_id && !payload.student_id) {
          utils.showNotification('Select exactly one target: classroom or student.', 'warning');
          return;
        }
        if (payload.classroom_id && payload.student_id) {
          utils.showNotification('Choose either classroom or student target, not both.', 'warning');
          return;
        }

        await api.teacher.createAssignment(payload);
        notifyTeacherSuccess('assignmentCreateSuccess');
        assignmentForm.reset();
        await loadAssignments();
      } catch (error) {
        console.error(error);
        notifyTeacherError('assignmentCreateFailed', error);
      } finally {
        clearBusy(btn);
      }
    });
  }

  const assignmentsBody = document.getElementById('assignmentsTableBody');
  if (assignmentsBody) {
    assignmentsBody.addEventListener('click', async evt => {
      const saveDueBtn = evt.target.closest('.assignment-save-due');
      if (saveDueBtn) {
        const assignmentId = Number(saveDueBtn.dataset.id);
        if (!assignmentId) return;

        const input = assignmentsBody.querySelector(`.assignment-due-input[data-id="${assignmentId}"]`);
        const raw = String(input?.value || '').trim();
        const dueIso = raw ? new Date(raw).toISOString() : null;

        try {
          await api.teacher.updateAssignment(assignmentId, { due_at: dueIso });
          utils.showNotification('Assignment due date/time updated.', 'success');
          await loadAssignments();
        } catch (error) {
          console.error(error);
          utils.showNotification(error.message || 'Failed to update due date/time', 'error');
        }
        return;
      }

      const clearDueBtn = evt.target.closest('.assignment-clear-due');
      if (clearDueBtn) {
        const assignmentId = Number(clearDueBtn.dataset.id);
        if (!assignmentId) return;

        try {
          await api.teacher.updateAssignment(assignmentId, { due_at: null });
          utils.showNotification('Assignment due date/time cleared.', 'success');
          await loadAssignments();
        } catch (error) {
          console.error(error);
          utils.showNotification(error.message || 'Failed to clear due date/time', 'error');
        }
        return;
      }

      const deleteBtn = evt.target.closest('.assignment-delete');
      if (!deleteBtn) return;
      const assignmentId = Number(deleteBtn.dataset.id);
      if (!assignmentId) return;

      if (!window.confirm('Delete this assignment permanently? This cannot be undone.')) return;

      try {
        await api.teacher.deleteAssignment(assignmentId);
        utils.showNotification('Assignment deleted.', 'success');
        await loadAssignments();
      } catch (error) {
        console.error(error);
        utils.showNotification(error.message || 'Failed to delete assignment', 'error');
      }
    });
  }

  // DOM Elements for Generation (safely inside the function)
  const generateBtn = document.getElementById('generate-btn');
  const questionsContainer = document.getElementById('generated-questions-container');
  const loadingIndicator = document.getElementById('loading-indicator');
  const progressText = document.getElementById('loading-progress-text');

  // Make sure the button actually exists on the page before adding the listener
  if (generateBtn) {
      generateBtn.addEventListener('click', async (e) => {
          e.preventDefault();

          const totalRequested = parseInt(document.getElementById('question-count').value, 10) || 10;
          const subject = document.getElementById('subject-select').value;
          const topic = document.getElementById('topic-input').value;
          const difficulty = document.getElementById('difficulty-select').value;
          const grade = document.getElementById('grade-select').value;

          const CHUNK_SIZE = 5; 
          let questionsFetched = 0;
          let allGeneratedQuestions = [];

          questionsContainer.innerHTML = ''; 
          generateBtn.disabled = true;
          loadingIndicator.style.display = 'block';

          try {
              while (questionsFetched < totalRequested) {
                  const countForThisChunk = Math.min(CHUNK_SIZE, totalRequested - questionsFetched);
                  progressText.innerText = `Generating ${questionsFetched} of ${totalRequested} questions...`;

                  // Hitting your custom HF Space via Render backend!
                  const result = await api.generateQuestions({
                      count: countForThisChunk,
                      subject,
                      topic,
                      difficulty,
                      grade
                  });

                  if (result.questions && result.questions.length > 0) {
                      const newQuestions = result.questions;
                      allGeneratedQuestions = allGeneratedQuestions.concat(newQuestions);
                      questionsFetched += newQuestions.length;

                      // Call the helper function
                      renderQuestionsChunkToDOM(newQuestions, questionsContainer);
                  } else {
                      console.warn("AI returned an empty chunk. Stopping early.");
                      break; 
                  }
              }
                
              progressText.innerText = `Success! Generated ${questionsFetched} questions.`;

          } catch (error) {
              console.error("Generation failed:", error);
              progressText.innerText = `Error generating questions: ${error.message}`;
          } finally {
              generateBtn.disabled = false;
              setTimeout(() => {
                  loadingIndicator.style.display = 'none';
              }, 2000);
          }
      });
  }

  bindTestActions();
  bindSidebarNav();
}

async function initTeacherDashboard() {
  const session = ensureTeacherSession();
  if (!session) return;

  setupProfileMenu(session);
  await initializeStemDropdowns();
  bindEvents();
  setAssessmentMode('topic');

  try {
    await loadAllTeacherData();
  } catch (error) {
    console.error(error);
    utils.showNotification(error.message || 'Failed to load teacher dashboard', 'error');
  }
}

document.addEventListener('DOMContentLoaded', initTeacherDashboard);

// Place this at the bottom of the file
function renderQuestionsChunkToDOM(questions, container) {
    questions.forEach((q, index) => {
        const qCard = document.createElement('div');
        qCard.className = 'question-card fade-in'; 
        
        const optionsHTML = q.options.map((opt, i) => {
            const isCorrect = i === q.correct_index;
            return `<li class="${isCorrect ? 'correct-option' : ''}">${opt}</li>`;
        }).join('');

        qCard.innerHTML = `
            <h4>${q.text}</h4>
            <ul>${optionsHTML}</ul>
            <div class="explanation">
                <strong>Explanation:</strong> ${q.explanation}
            </div>
        `;
        
        container.appendChild(qCard);
    });
}
