/**
 * Elevate Admin Control Center — admin.js
 * Phases 4-9: Training Monitor, Model Registry, MCQ Observability,
 * Audit Trail, User Management, School Hierarchy, UX Parity
 */
import { api } from './api.js';
import { config } from './config.js';

// =====================================================
// Auth & Session
// =====================================================
const SESSION_KEY = 'elevate_user_session';

function loadSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY) || sessionStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (_) { return null; }
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(SESSION_KEY);
}

function enforceAdmin() {
  const sess = loadSession();
  if (!sess?.user || !sess?.token) { window.location.replace('/index.html'); return null; }
  const role = String(sess.user.role || '').toLowerCase();
  if (role !== 'admin') {
    const pages = { teacher: 'teacher-dashboard.html', student: 'dashboard.html' };
    window.location.replace(pages[role] || '/index.html');
    return null;
  }
  return sess;
}

// =====================================================
// Toast Notifications
// =====================================================
let toastContainer = null;
function showToast(msg, type = 'info') {
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container';
    document.body.appendChild(toastContainer);
  }
  const icons = { success: 'check-circle', error: 'exclamation-circle', warning: 'exclamation-triangle', info: 'info-circle' };
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<i class="fas fa-${icons[type] || 'info-circle'}"></i><span>${msg}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// =====================================================
// Confirm Modal
// =====================================================
function showConfirm(title, msg) {
  return new Promise(resolve => {
    const overlay = document.getElementById('confirmModal');
    document.getElementById('confirmModalTitle').textContent = title;
    document.getElementById('confirmModalMessage').textContent = msg;
    overlay.classList.remove('hidden');

    const yes = document.getElementById('confirmModalYes');
    const no = document.getElementById('confirmModalNo');

    function cleanup(result) {
      overlay.classList.add('hidden');
      yes.removeEventListener('click', onYes);
      no.removeEventListener('click', onNo);
      resolve(result);
    }
    function onYes() { cleanup(true); }
    function onNo() { cleanup(false); }
    yes.addEventListener('click', onYes);
    no.addEventListener('click', onNo);
  });
}

// =====================================================
// Pagination helper
// =====================================================
function renderPagination(container, currentPage, totalPages, onPage) {
  container.innerHTML = '';
  if (totalPages <= 1) return;

  const prev = document.createElement('button');
  prev.className = 'page-btn';
  prev.innerHTML = '<i class="fas fa-chevron-left"></i>';
  prev.disabled = currentPage <= 1;
  prev.addEventListener('click', () => onPage(currentPage - 1));
  container.appendChild(prev);

  const info = document.createElement('span');
  info.className = 'page-info';
  info.textContent = `${currentPage} / ${totalPages}`;
  container.appendChild(info);

  const next = document.createElement('button');
  next.className = 'page-btn';
  next.innerHTML = '<i class="fas fa-chevron-right"></i>';
  next.disabled = currentPage >= totalPages;
  next.addEventListener('click', () => onPage(currentPage + 1));
  container.appendChild(next);
}

// =====================================================
// Panel Navigation
// =====================================================
function loadPanelData(panelKey, force = false) {
  switch (panelKey) {
    case 'dashboard':
      return loadDashboardStats();
    case 'users':
      return loadUsers(force ? 1 : usersPage || 1);
    case 'schools':
      return loadSchools();
    case 'test-results':
      return loadTestResults(force ? 1 : resPage || 1);
    case 'training':
      return loadTrainingJobs(force ? 1 : jobsPage || 1);
    case 'model-registry':
      return Promise.all([
        loadModelVersions(force ? 1 : versionsPage || 1),
        loadModelRegistrySummary(),
      ]);
    case 'mcq-obs':
      return loadMcqObservability();
    case 'audit':
      return loadAuditLogs(force ? 1 : auditPage || 1);
    default:
      return Promise.resolve();
  }
}

function activatePanel(panelId, forceLoad = false) {
  document.querySelectorAll('.admin-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const panel = document.getElementById(`panel-${panelId}`);
  const nav = document.getElementById(`nav-${panelId}`);
  panel?.classList.add('active');
  nav?.classList.add('active');

  const label = nav?.querySelector('span')?.textContent;
  if (label) {
    document.getElementById('topbarTitle').textContent = label;
  }

  document.getElementById('adminSidebar').classList.remove('open');
  loadPanelData(panelId, forceLoad).catch(err => {
    showToast(`Failed to load panel data: ${err.message}`, 'error');
  });
}

function setupPanelNav() {
  const navItems = document.querySelectorAll('.nav-item[data-panel]');
  navItems.forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const panelId = item.dataset.panel;
      activatePanel(panelId);
    });
  });
}

// Quick action buttons on dashboard
function setupQuickActions() {
  document.getElementById('qaUserManagement')?.addEventListener('click', () => activatePanel('users'));
  document.getElementById('qaTestResults')?.addEventListener('click', () => activatePanel('test-results'));
  document.getElementById('qaTriggerTraining')?.addEventListener('click', () => activatePanel('training'));
  document.getElementById('qaAuditLogs')?.addEventListener('click', () => activatePanel('audit'));
}

// =====================================================
// Server health
// =====================================================
async function checkServer() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('serverStatusText');
  const pre = document.getElementById('serverInfoPre');
  try {
    // Dynamically grab the backend URL from your config
    const baseUrl = config.API_BASE_URL || config.API_URL || config.BASE_URL || '';
    
    const res = await fetch(`${baseUrl}/health`);
    const data = await res.json().catch(() => ({}));
    
    dot.className = 'status-dot online';
    text.textContent = data.status || 'Online';
    if (pre) pre.textContent = JSON.stringify(data, null, 2);
  } catch {
    dot.className = 'status-dot offline';
    text.textContent = 'Offline';
    if (pre) pre.textContent = 'Backend unreachable.';
  }
}

// =====================================================
// Dashboard Stats
// =====================================================
async function loadDashboardStats() {
  try {
    const data = await api.admin.getStats();
    const setStat = (id, value) => {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    };
    setStat('statUsersVal', data.users ?? '—');
    setStat('statQuestionsVal', data.questions ?? '—');
    setStat('statEmotionsVal', data.emotion_logs ?? '—');
  } catch (err) {
    showToast('Failed to load stats: ' + err.message, 'error');
  }
}

// =====================================================
// Users Panel
// =====================================================
let usersPage = 1;
const usersPerPage = 15;
let currentUsersList = []; // Store users to easily find them for the edit modal
let allSchools = [];

async function loadUsers(page = 1) {
  usersPage = page;
  const searchInput = document.getElementById('userSearch');
  const roleFilter = document.getElementById('userRoleFilter');
  
  const search = searchInput ? searchInput.value.trim() : '';
  const role = roleFilter ? roleFilter.value : '';
  
  const tbody = document.getElementById('usersTableBody');
  const adminBody = document.getElementById('adminProfileBody');
  const currentUser = loadSession().user;

  if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="table-empty"><div class="spinner"></div></td></tr>`;

  // 1. Render Admin Profile Top Section
  if (adminBody && currentUser) {
      adminBody.innerHTML = `
          <tr>
              <td>
                  <div class="user-cell" style="display:flex;align-items:center;gap:12px;">
                      <div class="user-avatar" style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg, #6366f1, #8b5cf6);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:bold;">${(currentUser.name || 'A').charAt(0).toUpperCase()}</div>
                      <div class="user-details" style="display:flex;flex-direction:column;">
                          <span class="user-name" style="font-weight:600;color:#e2e8f0">${esc(currentUser.name)}</span>
                      </div>
                  </div>
              </td>
              <td style="color:#94a3b8">${esc(currentUser.email)}</td>
              <td><span class="badge badge-purple">Admin</span></td>
              <td><span class="badge badge-success">Active</span></td>
          </tr>
      `;
  }

  try {
    const data = await api.admin.listUsers({ page, per_page: usersPerPage, search, role });
    currentUsersList = data.items || [];

    if (!currentUsersList.length) {
      if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="table-empty">No users found.</td></tr>`;
    } else {
      // 2. Filter out admins from the bottom list
      const editableUsers = currentUsersList.filter(u => u.role !== 'admin');

      if (!editableUsers.length && tbody) {
           tbody.innerHTML = `<tr><td colspan="6" class="table-empty">No teachers or students found.</td></tr>`;
      } else if (tbody) {
          tbody.innerHTML = editableUsers.map(u => {
            const roleBadge = u.role === 'teacher' ? 'badge-teal' : 'badge-blue';
            const statusBadge = u.is_disabled ? '<span class="badge badge-red">Disabled</span>' : '<span class="badge badge-green">Active</span>';
            const school = allSchools.find(s => s.id === u.school_id);

            return `<tr>
              <td>
                <div class="user-cell" style="display:flex;align-items:center;gap:12px;">
                    <div class="user-avatar" style="width:32px;height:32px;border-radius:50%;background:#3b82f6;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:bold;">${(u.name || '?').charAt(0).toUpperCase()}</div>
                    <div class="user-details" style="display:flex;flex-direction:column;">
                        <span class="user-name" style="font-weight:600;color:#e2e8f0">${esc(u.name)}</span>
                        <span class="user-email" style="font-size:0.75rem;color:#94a3b8;">${esc(u.email)}</span>
                    </div>
                </div>
              </td>
              <td><span class="badge ${roleBadge}">${u.role}</span></td>
              <td>${esc(u.grade || '—')}</td>
              <td>${school ? esc(school.name) : '<span style="color:#6b7280">Unassigned</span>'}</td>
              <td>${statusBadge}</td>
              <td class="text-right" style="white-space:nowrap;">
                <button class="action-btn action-btn-primary edit-user-btn" data-id="${u.id}"><i class="fas fa-edit"></i> Edit</button>
                ${u.is_disabled
                  ? `<button class="action-btn action-btn-success enable-user-btn" data-id="${u.id}">Enable</button>`
                  : `<button class="action-btn action-btn-danger disable-user-btn" data-id="${u.id}">Disable</button>`}
              </td>
            </tr>`;
          }).join('');

          // Bind row events
          tbody.querySelectorAll('.edit-user-btn').forEach(btn => btn.addEventListener('click', () => openUserEditModal(parseInt(btn.dataset.id))));
          tbody.querySelectorAll('.disable-user-btn').forEach(btn => btn.addEventListener('click', () => quickDisableUser(parseInt(btn.dataset.id))));
          tbody.querySelectorAll('.enable-user-btn').forEach(btn => btn.addEventListener('click', () => quickEnableUser(parseInt(btn.dataset.id))));
      }
    }
    const totalPages = Math.ceil((data.total || 0) / usersPerPage);
    renderPagination(document.getElementById('usersPagination'), page, totalPages, loadUsers);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="table-empty">Error: ${esc(err.message)}</td></tr>`;
  }
}

function openUserEditModal(userId) {
  const u = currentUsersList.find(x => x.id === userId);
  if (!u) return;
  
  document.getElementById('editUserId').value = u.id;
  document.getElementById('editUserName').value = u.name;
  document.getElementById('editUserEmail').value = u.email;
  document.getElementById('editUserRole').value = u.role;
  
  // Populate school select
  const schoolSel = document.getElementById('editUserSchool');
  if (schoolSel) {
    schoolSel.innerHTML = '<option value="">— Unassigned —</option>' +
      allSchools.map(s => `<option value="${s.id}" ${u.school_id === s.id ? 'selected' : ''}>${esc(s.name)}</option>`).join('');
  }

  document.getElementById('userEditModal').classList.remove('hidden');
}

async function quickDisableUser(userId) {
  if (!await showConfirm('Disable User', 'Are you sure you want to disable this user?')) return;
  try {
    await api.admin.disableUser(userId, 'Admin action');
    showToast('User disabled', 'success');
    loadUsers(usersPage);
  } catch (err) { showToast(err.message, 'error'); }
}

async function quickEnableUser(userId) {
  try {
    await api.admin.enableUser(userId);
    showToast('User enabled', 'success');
    loadUsers(usersPage);
  } catch (err) { showToast(err.message, 'error'); }
}

function setupUsersPanel() {
  // Bind Search Input
  const searchInput = document.getElementById('userSearch');
  if (searchInput) {
      searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') loadUsers(1); });
  }

  // Bind Role Filter
  const roleFilter = document.getElementById('userRoleFilter');
  if (roleFilter) {
      roleFilter.addEventListener('change', () => loadUsers(1));
  }

  // Edit Modal Events
  const modal = document.getElementById('userEditModal');
  document.getElementById('closeUserEditModal')?.addEventListener('click', () => modal.classList.add('hidden'));
  document.getElementById('cancelUserEditBtn')?.addEventListener('click', () => modal.classList.add('hidden'));

  // Save changes from the Edit Modal
  document.getElementById('saveUserEditBtn')?.addEventListener('click', async () => {
    const userId = document.getElementById('editUserId').value;
    const role = document.getElementById('editUserRole').value;
    const school_id = document.getElementById('editUserSchool').value || null;
    
    try {
      await api.admin.updateUser(userId, {
        role,
        school_id: school_id ? parseInt(school_id) : null,
      });
      showToast('User updated successfully', 'success');
      modal.classList.add('hidden');
      loadUsers(usersPage);
    } catch (err) {
      showToast(err.message || 'Failed to update user', 'error');
    }
  });

  // --- Import / Add Users Logic ---
  const importModal = document.getElementById('importUsersModal');
  const importFeedback = document.getElementById('importFeedback');
  const tabBulkBtn = document.getElementById('tabBulkBtn');
  const tabSingleBtn = document.getElementById('tabSingleBtn');
  const tabBulkContent = document.getElementById('tabBulkContent');
  const tabSingleContent = document.getElementById('tabSingleContent');
  const warningBox = document.getElementById('singleAddWarning');

  document.getElementById('openImportModalBtn')?.addEventListener('click', () => {
    importModal.classList.remove('hidden');
    importFeedback.classList.add('hidden');
    warningBox.classList.add('hidden');
  });

  document.getElementById('closeImportModal')?.addEventListener('click', () => importModal.classList.add('hidden'));

  // Tab Switching Logic
  tabBulkBtn?.addEventListener('click', () => {
    tabBulkBtn.classList.add('active');
    tabSingleBtn.classList.remove('active');
    tabBulkContent.classList.remove('hidden');
    tabSingleContent.classList.add('hidden');
    importFeedback.classList.add('hidden');
  });

  tabSingleBtn?.addEventListener('click', () => {
    tabSingleBtn.classList.add('active');
    tabBulkBtn.classList.remove('active');
    tabSingleContent.classList.remove('hidden');
    tabBulkContent.classList.add('hidden');
    importFeedback.classList.add('hidden');
  });

  // 1. Download Template (100% Client-Side, No Backend Needed!)
  document.getElementById('downloadCsvTemplateBtn')?.addEventListener('click', () => {
    try {
      // 1. We write the exact CSV text directly in the browser
      const csvText = "Name,Email,Role,Grade\nJohn Doe,john.doe@example.com,student,College\nJane Smith,jane.smith@example.com,teacher,\n";

      // 2. Use your existing utility function to force the browser to download it
      downloadText(csvText, 'elevate_users_template.csv', 'text/csv');
      
      showToast('Template downloaded successfully!', 'success');
    } catch (err) {
      console.error(err);
      showToast('Failed to download template.', 'error');
    }
  });

  // 2. Bulk Upload
  document.getElementById('uploadCsvBtn')?.addEventListener('click', async () => {
    const fileInput = document.getElementById('csvFileInput');
    if (!fileInput.files[0]) return showToast('Please select a CSV file first.', 'warning');
    
    // --> NEW CONFIRMATION DIALOG <--
    if (!await showConfirm('Confirm Bulk Assignment', 'Are you sure you want to process this list and link matching users to your school?')) return;

    const btn = document.getElementById('uploadCsvBtn');
    btn.innerHTML = '<div class="spinner"></div> Processing...';
    btn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      const res = await api.admin.bulkImportUsers(formData);
      
      importFeedback.className = '';
      importFeedback.style = 'margin-top:16px; padding:12px; border-radius:8px; text-align:center; font-size:0.875rem; font-weight:500; background:rgba(16,185,129,0.1); color:#34d399; border:1px solid rgba(16,185,129,0.2);';
      importFeedback.textContent = `Success! Linked ${res.updated} users to your school. (${res.not_found} users were not found in the database).`;
      showToast('Bulk processing complete', 'success');
      loadUsers(1);
    } catch (err) {
      importFeedback.className = '';
      importFeedback.style = 'margin-top:16px; padding:12px; border-radius:8px; text-align:center; font-size:0.875rem; font-weight:500; background:rgba(239,68,68,0.1); color:#f87171; border:1px solid rgba(239,68,68,0.2);';
      importFeedback.textContent = err.error || err.message || 'Upload failed.';
    } finally {
      btn.innerHTML = '<i class="fas fa-upload"></i> Process Upload';
      btn.disabled = false;
    }
  });

  // 3. Single User Search / Add
  document.getElementById('singleAddBtn')?.addEventListener('click', async () => {
    const email = document.getElementById('singleAddEmail').value.trim();
    const name = document.getElementById('singleAddName').value.trim();
    const role = document.getElementById('singleAddRole').value;
    
    if (!email || !name) return showToast('Email and Name are required', 'warning');
    
    // --> NEW CONFIRMATION DIALOG <--
    if (!await showConfirm('Confirm User Assignment', `Are you sure you want to search and add ${name} to your school?`)) return;

    const btn = document.getElementById('singleAddBtn');
    btn.innerHTML = '<div class="spinner"></div> Searching...';
    btn.disabled = true;
    warningBox.classList.add('hidden');
    importFeedback.classList.add('hidden');
    
    try {
      const res = await api.admin.singleAddUser({ email, name, role });
      importFeedback.className = '';
      importFeedback.style = 'margin-top:16px; padding:12px; border-radius:8px; text-align:center; font-size:0.875rem; font-weight:500; background:rgba(16,185,129,0.1); color:#34d399; border:1px solid rgba(16,185,129,0.2);';
      importFeedback.textContent = res.message;
      showToast('User linked successfully!', 'success');
      loadUsers(1);
    } catch (err) {
      console.error("Single Add Error:", err);
      // Safely extract the JSON payload attached to the Error object by api.js
      const payload = err.payload || {};
      
      if (payload.status === 'similar_found') {
        // Show the intelligent yellow warning box
        warningBox.innerHTML = `<strong>Wait!</strong><br/>${payload.error}<br/><br/><em>${payload.suggestion}</em>`;
        warningBox.classList.remove('hidden');
      } else {
        // Show standard red error using the exact message from the backend
        importFeedback.className = '';
        importFeedback.style = 'margin-top:16px; padding:12px; border-radius:8px; text-align:center; font-size:0.875rem; font-weight:500; background:rgba(239,68,68,0.1); color:#f87171; border:1px solid rgba(239,68,68,0.2);';
        importFeedback.textContent = err.serverMessage || payload.error || err.message || 'Failed to process user.';
      }
    } finally {
      btn.innerHTML = '<i class="fas fa-user-plus"></i> Add User';
      btn.disabled = false;
    }
  });
}
// =====================================================
// (End of Users Panel)
// =====================================================

// function openUserEditModal(userId, items) {
//   const u = items.find(x => x.id === userId);
//   if (!u) return;
//   editingUserId = userId;
//   editingUserIsDisabled = u.is_disabled;
//   document.getElementById('editUserRole').value = u.role;

//   // Populate school select
//   const schoolSel = document.getElementById('editUserSchool');
//   schoolSel.innerHTML = '<option value="">— No School —</option>' +
//     allSchools.map(s => `<option value="${s.id}" ${u.school_id === s.id ? 'selected' : ''}>${esc(s.name)}</option>`).join('');

//   const toggleBtn = document.getElementById('userModalToggleDisable');
//   toggleBtn.textContent = u.is_disabled ? 'Enable User' : 'Disable User';
//   toggleBtn.className = u.is_disabled ? 'btn btn-success' : 'btn btn-danger';

//   document.getElementById('userModalTitle').textContent = `Edit User: ${u.name}`;
//   document.getElementById('userModalFeedback').textContent = '';
//   document.getElementById('userModal').classList.remove('hidden');
// }

// async function quickDisableUser(userId) {
//   if (!await showConfirm('Disable User', 'Are you sure you want to disable this user?')) return;
//   try {
//     await api.admin.disableUser(userId, 'Admin action');
//     showToast('User disabled', 'success');
//     loadUsers(usersPage);
//   } catch (err) { showToast(err.message, 'error'); }
// }

// async function quickEnableUser(userId) {
//   try {
//     await api.admin.enableUser(userId);
//     showToast('User enabled', 'success');
//     loadUsers(usersPage);
//   } catch (err) { showToast(err.message, 'error'); }
// }

// function setupUsersPanel() {
//   document.getElementById('usersSearchBtn').addEventListener('click', () => loadUsers(1));
//   document.getElementById('userSearch').addEventListener('keydown', e => { if (e.key === 'Enter') loadUsers(1); });

//   document.getElementById('userModalClose').addEventListener('click', () => document.getElementById('userModal').classList.add('hidden'));
//   document.getElementById('userModalCancel').addEventListener('click', () => document.getElementById('userModal').classList.add('hidden'));

//   document.getElementById('userModalSave').addEventListener('click', async () => {
//     if (!editingUserId) return;
//     const role = document.getElementById('editUserRole').value;
//     const school_id = document.getElementById('editUserSchool').value || null;
//     const feedback = document.getElementById('userModalFeedback');
//     feedback.textContent = 'Saving...';
//     try {
//       await api.admin.updateUser(editingUserId, {
//         role,
//         school_id: school_id ? parseInt(school_id) : null,
//       });
//       feedback.style.color = '#34d399';
//       feedback.textContent = 'Saved!';
//       showToast('User updated', 'success');
//       document.getElementById('userModal').classList.add('hidden');
//       loadUsers(usersPage);
//     } catch (err) {
//       feedback.style.color = '#f87171';
//       feedback.textContent = err.message;
//     }
//   });

//   document.getElementById('userModalToggleDisable').addEventListener('click', async () => {
//     if (!editingUserId) return;
//     const reason = document.getElementById('editUserDisableReason').value.trim();
//     const feedback = document.getElementById('userModalFeedback');
//     try {
//       if (editingUserIsDisabled) {
//         await api.admin.enableUser(editingUserId);
//         showToast('User enabled', 'success');
//       } else {
//         await api.admin.disableUser(editingUserId, reason || 'Admin action');
//         showToast('User disabled', 'warning');
//       }
//       document.getElementById('userModal').classList.add('hidden');
//       loadUsers(usersPage);
//     } catch (err) {
//       feedback.style.color = '#f87171';
//       feedback.textContent = err.message;
//     }
//   });
// }

// =====================================================
// Schools Panel
// =====================================================
let schoolsPage = 1;
let showingHierarchy = false;

async function loadSchoolsHierarchy() {
  const container = document.getElementById('hierarchyTreeContainer');
  container.innerHTML = '<div class="table-empty"><div class="spinner"></div></div>';
  try {
    const data = await api.admin.getSchoolsHierarchy();
    const items = data.items || [];
    if (!items.length) { container.innerHTML = '<div class="table-empty">No schools found.</div>'; return; }
    container.innerHTML = items.map(school => `
      <div class="hierarchy-school">
        <div class="hierarchy-school-header" data-school-id="${school.id}">
          <div class="hierarchy-school-name">
            <i class="fas fa-school" style="color:#818cf8;font-size:14px"></i>
            ${esc(school.name)}
          </div>
          <div class="hierarchy-school-meta">
            <span><i class="fas fa-chalkboard-teacher" style="color:#6b7280"></i> ${school.teacher_count} teachers</span>
            <span><i class="fas fa-user-graduate" style="color:#6b7280"></i> ${school.student_count} students</span>
            <i class="fas fa-chevron-down" style="color:#6b7280;margin-left:8px;transition:transform 0.2s"></i>
          </div>
        </div>
        <div class="hierarchy-teachers" id="hteach-${school.id}" data-loaded="0">
          <div style="color:#6b7280;padding:8px 0;font-size:0.8rem">Click to load teacher and student hierarchy.</div>
        </div>
      </div>
    `).join('');

    // Toggle expand/collapse
    container.querySelectorAll('.hierarchy-school-header').forEach(header => {
      header.addEventListener('click', async () => {
        const id = header.dataset.schoolId;
        const teachers = document.getElementById(`hteach-${id}`);
        const chevron = header.querySelector('.fa-chevron-down');
        teachers.classList.toggle('expanded');
        if (chevron) chevron.style.transform = teachers.classList.contains('expanded') ? 'rotate(180deg)' : '';

        if (!teachers.classList.contains('expanded') || teachers.dataset.loaded === '1') {
          return;
        }

        teachers.innerHTML = '<div class="table-empty"><div class="spinner"></div></div>';
        try {
          const detail = await api.admin.getSchoolHierarchyDetail(id);
          const teacherItems = detail.teachers || [];
          if (!teacherItems.length) {
            teachers.innerHTML = '<div style="color:#4b5563;padding:8px 0;font-size:0.8rem">No teachers assigned</div>';
            teachers.dataset.loaded = '1';
            return;
          }

          teachers.innerHTML = teacherItems.map(t => {
            const classroomRows = (t.classrooms || []).map(cls => {
              const studentRows = (cls.students || []).map(st => `<span class="hierarchy-student-pill">${esc(st.name || st.email || 'Student')}</span>`).join('');
              return `
                <div class="hierarchy-classroom-row">
                  <div class="hierarchy-classroom-head">
                    <span class="hierarchy-classroom-name">${esc(cls.name || 'Classroom')}</span>
                    <span class="hierarchy-classroom-meta">${(cls.students || []).length} students</span>
                  </div>
                  <div class="hierarchy-student-list">${studentRows || '<span class="hierarchy-empty">No students enrolled</span>'}</div>
                </div>
              `;
            }).join('');

            return `
              <div class="hierarchy-teacher">
                <i class="fas fa-chalkboard-teacher" style="color:#6366f1;font-size:12px"></i>
                <div style="flex:1">
                  <div style="font-weight:600;font-size:0.85rem;color:#c7d2fe">${esc(t.name)}</div>
                  <div style="font-size:0.74rem;color:#6b7280;margin-bottom:8px">${esc(t.email)}</div>
                  <div class="hierarchy-classroom-list">${classroomRows || '<span class="hierarchy-empty">No classrooms</span>'}</div>
                </div>
              </div>
            `;
          }).join('');
          teachers.dataset.loaded = '1';
        } catch (detailErr) {
          teachers.innerHTML = `<div class="table-empty">Error: ${esc(detailErr.message)}</div>`;
        }
      });
    });
  } catch (err) {
    container.innerHTML = `<div class="table-empty">Error: ${esc(err.message)}</div>`;
  }
}

async function loadSchools() {
  const tbody = document.getElementById('schoolsTableBody');
  tbody.innerHTML = `<tr><td colspan="7" class="table-empty"><div class="spinner"></div></td></tr>`;
  try {
    const data = await api.admin.getSchoolsHierarchy();
    allSchools = data.items || [];
    if (!allSchools.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="table-empty">No schools found.</td></tr>`;
      return;
    }
    tbody.innerHTML = allSchools.map(s => `<tr>
      <td>${s.id}</td>
      <td><strong>${esc(s.name)}</strong></td>
      <td style="color:#6b7280">${esc(s.slug || '—')}</td>
      <td>${s.teacher_count ?? 0}</td>
      <td>${s.student_count ?? 0}</td>
      <td style="color:#6b7280;font-size:0.78rem">${s.created_at ? new Date(s.created_at).toLocaleDateString() : '—'}</td>
      <td>
        <button class="action-btn action-btn-danger delete-school-btn" data-id="${s.id}">Delete</button>
      </td>
    </tr>`).join('');

    tbody.querySelectorAll('.delete-school-btn').forEach(btn => btn.addEventListener('click', async () => {
      const id = parseInt(btn.dataset.id);
      if (!await showConfirm('Delete School', 'This will remove the school. Users will lose their school association.')) return;
      try {
        await api.admin.deleteSchool(id);
        showToast('School deleted', 'warning');
        loadSchools();
      } catch (err) { showToast(err.message, 'error'); }
    }));

    // Update user school filter dropdown
    const schoolSel = document.getElementById('userSchoolFilter');
    if (schoolSel) {
      schoolSel.innerHTML = '<option value="">All Schools</option>' +
        allSchools.map(s => `<option value="${s.id}">${esc(s.name)}</option>`).join('');
    }

    const resultsSchoolSel = document.getElementById('resultsSchoolFilter');
    if (resultsSchoolSel) {
      resultsSchoolSel.innerHTML = '<option value="">All Schools</option>' +
        allSchools.map(s => `<option value="${s.id}">${esc(s.name)}</option>`).join('');
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="table-empty">Error: ${esc(err.message)}</td></tr>`;
  }
}

function setupSchoolsPanel() {
  document.getElementById('toggleSchoolViewBtn').addEventListener('click', () => {
    showingHierarchy = !showingHierarchy;
    document.getElementById('schoolListView').classList.toggle('hidden', showingHierarchy);
    document.getElementById('schoolHierarchyView').classList.toggle('hidden', !showingHierarchy);
    document.getElementById('toggleSchoolViewBtn').innerHTML = showingHierarchy
      ? '<i class="fas fa-table"></i> List View'
      : '<i class="fas fa-sitemap"></i> Hierarchy View';
    if (showingHierarchy) loadSchoolsHierarchy();
  });

  document.getElementById('addSchoolBtn').addEventListener('click', () => {
    document.getElementById('addSchoolModal').classList.remove('hidden');
    document.getElementById('addSchoolFeedback').textContent = '';
  });

  document.getElementById('addSchoolModalClose').addEventListener('click', () => document.getElementById('addSchoolModal').classList.add('hidden'));
  document.getElementById('addSchoolCancel').addEventListener('click', () => document.getElementById('addSchoolModal').classList.add('hidden'));

  document.getElementById('addSchoolSave').addEventListener('click', async () => {
    const name = document.getElementById('newSchoolName').value.trim();
    const slug = document.getElementById('newSchoolSlug').value.trim();
    const fb = document.getElementById('addSchoolFeedback');
    if (!name || !slug) { fb.textContent = 'Name and slug required.'; return; }
    try {
      fb.textContent = 'Creating...';
      await api.admin.createSchool({ name, slug });
      fb.textContent = '';
      showToast('School created!', 'success');
      document.getElementById('addSchoolModal').classList.add('hidden');
      document.getElementById('newSchoolName').value = '';
      document.getElementById('newSchoolSlug').value = '';
      loadSchools();
    } catch (err) { fb.textContent = err.message; }
  });

  // Auto-slug from name
  document.getElementById('newSchoolName').addEventListener('input', e => {
    const slug = e.target.value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    document.getElementById('newSchoolSlug').value = slug;
  });
}

// =====================================================
// Test Results Panel
// =====================================================
let resPage = 1;
const resPerPage = 12;

async function loadTestResults(page = 1) {
  resPage = page;
  const minScoreValue = parseFloat(document.getElementById('resultsMinScore')?.value);
  const maxScoreValue = parseFloat(document.getElementById('resultsMaxScore')?.value);
  
  // Get the current admin's school ID
  const currentUser = loadSession().user;

  const schoolFilterValue = document.getElementById('resultsSchoolFilter')?.value;
  const params = {
    page, per_page: resPerPage,
    subject: document.getElementById('resultsSubjectFilter')?.value.trim() || '',
    email: document.getElementById('resultsEmailFilter')?.value.trim() || '',
    status: document.getElementById('resultsStatusFilter')?.value || '',
    school_id: schoolFilterValue || currentUser.school_id || '',
    start: document.getElementById('resultsDateFrom')?.value || '',
    end: document.getElementById('resultsDateTo')?.value || '',
    min_score: Number.isFinite(minScoreValue) ? minScoreValue : 0,
    max_score: Number.isFinite(maxScoreValue) ? maxScoreValue : 100,
  };
  
  const tbody = document.getElementById('resultsTableBody');
  tbody.innerHTML = `<tr><td colspan="9" class="table-empty"><div class="spinner"></div></td></tr>`;
  try {
    const data = await api.admin.listTestResults(params);
    const items = data.items || [];
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="table-empty">No results found.</td></tr>`;
    } else {
      tbody.innerHTML = items.map(r => {
        const pct = r.score_pct ?? 0;
        const scoreColor = pct >= 80 ? '#34d399' : pct >= 60 ? '#fbbf24' : '#f87171';
        const statusBadge = r.status === 'completed' ? 'badge-green' : r.status === 'in_progress' ? 'badge-blue' : 'badge-gray';
        const canViewHistory = Number.isFinite(Number(r.user?.id));
        return `<tr>
          <td>${r.id}</td>
          <td>
            <div style="font-weight:600">${esc(r.user?.name || 'Unknown')}</div>
            <div style="font-size:0.74rem;color:#6b7280">${esc(r.user?.email || '')}</div>
          </td>
          <td>${esc(r.subject)}</td>
          <td>
            <span style="color:${scoreColor};font-weight:700">${pct}%</span>
            <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%;background:${scoreColor}"></div></div>
          </td>
          <td>${r.correct_answers}/${r.total_questions}</td>
          <td>${r.avg_time_per_question ? r.avg_time_per_question.toFixed(1) + 's' : '—'}</td>
          <td><span class="badge ${statusBadge}">${r.status}</span></td>
          <td style="color:#6b7280;font-size:0.78rem">${r.started_at ? new Date(r.started_at).toLocaleDateString() : '—'}</td>
          <td>
            <button class="action-btn action-btn-primary view-result-btn" data-id="${r.id}">View</button>
            <button class="action-btn action-btn-primary view-history-detail-btn" data-user-id="${r.user?.id || ''}" ${canViewHistory ? '' : 'disabled'}>History</button>
          </td>
        </tr>`;
      }).join('');
      tbody.querySelectorAll('.view-result-btn').forEach(btn => btn.addEventListener('click', () => showTestDetail(parseInt(btn.dataset.id))));
      tbody.querySelectorAll('.view-history-detail-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const userId = parseInt(btn.dataset.userId, 10);
          if (!Number.isFinite(userId)) {
            showToast('History is unavailable for this row.', 'warning');
            return;
          }
          showUserHistory(userId);
        });
      });
    }
    const totalPages = Math.ceil((data.total || 0) / resPerPage);
    renderPagination(document.getElementById('resultsPagination'), page, totalPages, loadTestResults);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" class="table-empty">Error: ${esc(err.message)}</td></tr>`;
  }
}

let detailChartInstance = null;

async function showTestDetail(testResultId) {
  const modal = document.getElementById('testDetailModal');
  const body = document.getElementById('testDetailModalBody');
  document.getElementById('testDetailModalTitle').textContent = `Test Result #${testResultId}`;
  modal.classList.remove('hidden');
  body.innerHTML = '<div class="spinner"></div>';
  try {
    const data = await api.admin.getTestResultDetail(testResultId);
    const t = data.test;
    const answers = data.answers || [];
    const labels = answers.map((_, i) => `Q${i+1}`);
    const values = answers.map(a => a.is_correct ? 100 : 0);

    body.innerHTML = `
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px">
        <div><strong style="color:#94a3b8">Student:</strong> ${esc(t.user_name || '')} &lt;${esc(t.user_email || '')}&gt;</div>
        <div><strong style="color:#94a3b8">Subject:</strong> ${esc(t.subject)}</div>
        <div><strong style="color:#94a3b8">Score:</strong> <span style="color:${t.score_pct >= 80 ? '#34d399' : '#fbbf24'};font-weight:700">${t.score_pct}%</span></div>
        <div><strong style="color:#94a3b8">Answers:</strong> ${t.correct_answers}/${t.total_questions}</div>
        <div><strong style="color:#94a3b8">Avg Time:</strong> ${t.avg_time_per_question ? t.avg_time_per_question.toFixed(1) + 's' : '—'}</div>
        <div><strong style="color:#94a3b8">Status:</strong> ${t.status}</div>
      </div>
      <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:12px;margin-bottom:16px">
        <canvas id="detailBarChart" height="120"></canvas>
      </div>
      <div style="max-height:300px;overflow-y:auto">
        ${answers.map((a, i) => `
          <div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;gap:12px">
            <span style="color:#6b7280;width:30px;flex-shrink:0">Q${i+1}</span>
            <div style="flex:1;font-size:0.82rem">${esc(a.question_text || '—')}</div>
            <span style="color:${a.is_correct ? '#34d399' : '#f87171'};font-size:0.8rem;font-weight:600;flex-shrink:0">
              ${a.is_correct ? '✓ Correct' : '✗ Wrong'}
            </span>
            <span style="color:#6b7280;font-size:0.75rem;flex-shrink:0">${a.time_spent}s</span>
          </div>
        `).join('')}
      </div>
    `;

    setTimeout(() => {
      const ctx = document.getElementById('detailBarChart')?.getContext('2d');
      if (!ctx) return;
      if (detailChartInstance) { detailChartInstance.destroy(); detailChartInstance = null; }
      detailChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: values,
            backgroundColor: values.map(v => v === 100 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)'),
            borderRadius: 4,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            y: { suggestedMin: 0, suggestedMax: 100, ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { ticks: { color: '#6b7280' }, grid: { display: false } }
          }
        }
      });
    }, 50);
  } catch (err) {
    body.innerHTML = `<div style="color:#f87171">Error: ${esc(err.message)}</div>`;
  }
}

async function showUserHistory(userId) {
  const modal = document.getElementById('testDetailModal');
  const body = document.getElementById('testDetailModalBody');
  document.getElementById('testDetailModalTitle').textContent = `User History #${userId}`;
  modal.classList.remove('hidden');
  body.innerHTML = '<div class="spinner"></div>';
  try {
    const data = await api.admin.getTestResultHistory(userId);
    const items = data.items || [];
    if (!items.length) { body.innerHTML = '<div class="table-empty">No history found.</div>'; return; }

    const labels = items.map(i => i.started_at ? new Date(i.started_at).toLocaleDateString() : '');
    const scores = items.map(i => i.score_pct || 0);

    body.innerHTML = `
      <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:12px;margin-bottom:16px">
        <canvas id="historyLineChart" height="120"></canvas>
      </div>
      <div style="max-height:300px;overflow-y:auto">
        ${items.map(i => `
          <div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;gap:12px">
            <span style="font-size:0.78rem;color:#6b7280;width:90px;flex-shrink:0">${i.started_at ? new Date(i.started_at).toLocaleDateString() : ''}</span>
            <div style="flex:1">
              <span style="font-weight:600;color:#c7d2fe">${esc(i.subject)}</span>
              <span style="color:#6b7280;font-size:0.78rem;margin-left:8px">${i.correct_answers}/${i.total_questions}</span>
            </div>
            <span style="color:${i.score_pct >= 80 ? '#34d399' : i.score_pct >= 60 ? '#fbbf24' : '#f87171'};font-weight:700">${i.score_pct}%</span>
          </div>
        `).join('')}
      </div>
    `;

    setTimeout(() => {
      const ctx = document.getElementById('historyLineChart')?.getContext('2d');
      if (!ctx) return;
      if (detailChartInstance) { detailChartInstance.destroy(); detailChartInstance = null; }
      detailChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Score %',
            data: scores,
            borderColor: '#818cf8',
            backgroundColor: 'rgba(99,102,241,0.1)',
            fill: true, tension: 0.3, pointRadius: 4,
            pointBackgroundColor: '#818cf8',
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            y: { suggestedMin: 0, suggestedMax: 100, ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { ticks: { color: '#6b7280' }, grid: { display: false } }
          }
        }
      });
    }, 50);
  } catch (err) {
    body.innerHTML = `<div style="color:#f87171">Error: ${esc(err.message)}</div>`;
  }
}

function setupTestResultsPanel() {
  document.getElementById('resultsSearchBtn').addEventListener('click', () => loadTestResults(1));
  document.getElementById('resultsSubjectFilter')?.addEventListener('keydown', e => { if (e.key === 'Enter') loadTestResults(1); });
  document.getElementById('resultsEmailFilter')?.addEventListener('keydown', e => { if (e.key === 'Enter') loadTestResults(1); });
  ['resultsStatusFilter', 'resultsSchoolFilter', 'resultsDateFrom', 'resultsDateTo', 'resultsMinScore', 'resultsMaxScore'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => loadTestResults(1));
  });
  document.getElementById('exportResultsCsvBtn').addEventListener('click', async () => {
    const minScoreValue = parseFloat(document.getElementById('resultsMinScore').value);
    const maxScoreValue = parseFloat(document.getElementById('resultsMaxScore').value);
    try {
      const csv = await api.admin.exportTestResultsCsv({
        subject: document.getElementById('resultsSubjectFilter').value.trim(),
        email: document.getElementById('resultsEmailFilter').value.trim(),
        status: document.getElementById('resultsStatusFilter').value,
        school_id: document.getElementById('resultsSchoolFilter').value,
        start: document.getElementById('resultsDateFrom').value,
        end: document.getElementById('resultsDateTo').value,
        min_score: Number.isFinite(minScoreValue) ? minScoreValue : 0,
        max_score: Number.isFinite(maxScoreValue) ? maxScoreValue : 100,
      });
      downloadText(csv, `test_results_${today()}.csv`, 'text/csv');
      showToast('CSV downloaded', 'success');
    } catch (err) { showToast(err.message, 'error'); }
  });
  document.getElementById('testDetailModalClose').addEventListener('click', () => {
    document.getElementById('testDetailModal').classList.add('hidden');
  });
}

// =====================================================
// Training Jobs Panel
// =====================================================
let jobsPage = 1;
const jobsPerPage = 10;

async function loadTrainingJobs(page = 1) {
  jobsPage = page;
  const status = document.getElementById('jobStatusFilter').value;
  const tbody = document.getElementById('jobsTableBody');
  tbody.innerHTML = `<tr><td colspan="8" class="table-empty"><div class="spinner"></div></td></tr>`;
  try {
    const data = await api.admin.listTrainingJobs({ page, per_page: jobsPerPage, status, sync: true });
    const items = data.items || [];
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="table-empty">No training jobs found.</td></tr>`;
    } else {
      tbody.innerHTML = items.map(j => {
        const sbadge = j.status === 'succeeded' ? 'badge-green' : j.status === 'running' ? 'badge-blue' : j.status === 'failed' ? 'badge-red' : 'badge-gray';
        const dur = j.duration_ms ? `${(j.duration_ms/1000).toFixed(0)}s` : '—';
        const shortJobId = j.job_id ? (String(j.job_id).length > 16 ? `${String(j.job_id).slice(0, 16)}...` : String(j.job_id)) : '—';
        return `<tr>
          <td>${j.id}</td>
          <td><code style="font-size:0.72rem;color:#94a3b8">${esc(shortJobId)}</code></td>
          <td><span class="badge ${sbadge}">${j.status}</span></td>
          <td>${esc(j.source || '—')}</td>
          <td>${esc(j.triggered_by_name || `ID:${j.triggered_by}` || '—')}</td>
          <td>${dur}</td>
          <td style="font-size:0.78rem;color:#6b7280">${j.started_at ? new Date(j.started_at).toLocaleString() : '—'}</td>
          <td>
            <button class="action-btn action-btn-primary view-job-btn" data-id="${j.id}">Details</button>
            ${j.status === 'running' ? `<button class="action-btn action-btn-warning refresh-job-btn" data-job-id="${j.job_id}">Sync</button>` : ''}
          </td>
        </tr>`;
      }).join('');
      tbody.querySelectorAll('.view-job-btn').forEach(btn => btn.addEventListener('click', () => showJobDetail(parseInt(btn.dataset.id))));
      tbody.querySelectorAll('.refresh-job-btn').forEach(btn => btn.addEventListener('click', async () => {
        try {
          await api.admin.getTrainingStatus(btn.dataset.jobId);
          showToast('Job status synced', 'info');
          loadTrainingJobs(jobsPage);
        } catch (err) { showToast(err.message, 'error'); }
      }));
    }
    const totalPages = Math.ceil((data.total || 0) / jobsPerPage);
    renderPagination(document.getElementById('jobsPagination'), page, totalPages, loadTrainingJobs);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="table-empty">Error: ${esc(err.message)}</td></tr>`;
  }
}

async function showJobDetail(jobDbId) {
  const body = document.getElementById('jobDetailModalBody');
  document.getElementById('jobDetailModal').classList.remove('hidden');
  document.getElementById('jobDetailModalTitle').textContent = `Job #${jobDbId}`;
  body.innerHTML = '<div class="spinner"></div>';

  let j;
  try {
    const detail = await api.admin.getTrainingJob(jobDbId);
    j = detail.job;
  } catch (err) {
    body.innerHTML = `<div style="color:#f87171">Error: ${esc(err.message)}</div>`;
    return;
  }

  const shortTitle = j.job_id ? (String(j.job_id).length > 20 ? `${String(j.job_id).slice(0, 20)}...` : String(j.job_id)) : `#${jobDbId}`;
  document.getElementById('jobDetailModalTitle').textContent = `Job: ${shortTitle}`;
  
  body.innerHTML = `
    <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px">
      <div><strong style="color:#94a3b8">Status:</strong> <span class="badge badge-blue">${j.status}</span></div>
      <div><strong style="color:#94a3b8">Trigger Source:</strong> ${j.trigger_source || '—'}</div>
      <div><strong style="color:#94a3b8">Duration:</strong> ${j.duration_seconds ? j.duration_seconds + 's' : '—'}</div>
      <div><strong style="color:#94a3b8">Started:</strong> ${j.started_at ? new Date(j.started_at).toLocaleString() : '—'}</div>
      <div><strong style="color:#94a3b8">Finished:</strong> ${j.finished_at ? new Date(j.finished_at).toLocaleString() : '—'}</div>
    </div>
    
    ${j.error_message ? `<div style="color:#f87171;background:rgba(239,68,68,0.1);padding:10px 14px;border-radius:8px;margin-bottom:16px;border:1px solid rgba(239,68,68,0.2);">⚠ ${esc(j.error_message)}</div>` : ''}
    
    ${j.metrics && Object.keys(j.metrics).length ? `
      <div style="margin-bottom:16px">
        <strong style="color:#94a3b8;display:block;margin-bottom:6px;">Final Metrics:</strong>
        <pre class="info-pre">${JSON.stringify(j.metrics, null, 2)}</pre>
      </div>` : ''}
      
    ${j.logs ? `
      <div style="margin-bottom:12px">
        <strong style="color:#94a3b8;display:block;margin-bottom:6px;">Execution Logs:</strong>
        <pre class="info-pre">${esc(j.logs)}</pre>
      </div>` : ''}
  `;
}

function setupTrainingPanel() {
  document.getElementById('refreshJobsBtn').addEventListener('click', () => loadTrainingJobs(1));
  document.getElementById('jobStatusFilter').addEventListener('change', () => loadTrainingJobs(1));
  document.getElementById('jobDetailModalClose').addEventListener('click', () => document.getElementById('jobDetailModal').classList.add('hidden'));

  document.getElementById('triggerTrainingBtn').addEventListener('click', async () => {
    if (!await showConfirm('Trigger Training', 'This will start a strict HF training pipeline. Continue?')) return;
    const btn = document.getElementById('triggerTrainingBtn');
    btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Triggering...';
    try {
      const result = await api.admin.triggerTraining();
      showToast(`Training triggered! Job: ${result.job_id || 'started'}`, 'success');
      loadTrainingJobs(1);
    } catch (err) { showToast('Failed: ' + err.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = '<i class="fas fa-play-circle"></i> Trigger Training'; }
  });
}

// =====================================================
// Model Registry Panel
// =====================================================
let versionsPage = 1;
const versionsPerPage = 15;

async function loadModelVersions(page = 1) {
  versionsPage = page;
  const tbody = document.getElementById('versionsTableBody');
  tbody.innerHTML = `<tr><td colspan="8" class="table-empty"><div class="spinner"></div></td></tr>`;
  try {
    const data = await api.admin.listModelVersions({ page, per_page: versionsPerPage });
    const items = data.items || [];
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="table-empty">No model versions registered yet.</td></tr>`;
    } else {
      tbody.innerHTML = items.map(v => `<tr>
        <td>${v.id}</td>
        <td><strong>${esc(v.model_name)}</strong></td>
        <td><code style="color:#818cf8">${esc(v.version_tag)}</code></td>
        <td>${v.is_production ? '<span class="badge badge-green">✓ Production</span>' : '—'}</td>
        <td>${v.is_rollback_candidate ? '<span class="badge badge-orange">✓ Rollback</span>' : '—'}</td>
        <td style="font-size:0.75rem;color:#6b7280">${Object.keys(v.metrics || {}).length ? JSON.stringify(v.metrics).slice(0,60) + '…' : '—'}</td>
        <td style="font-size:0.78rem;color:#6b7280">${v.created_at ? new Date(v.created_at).toLocaleDateString() : '—'}</td>
        <td>
          ${!v.is_production ? `<button class="action-btn action-btn-success promote-version-btn" data-id="${v.id}">Promote</button>` : ''}
          ${!v.is_rollback_candidate ? `<button class="action-btn action-btn-warning rollback-version-btn" data-id="${v.id}">Set Rollback</button>` : ''}
        </td>
      </tr>`).join('');
      tbody.querySelectorAll('.promote-version-btn').forEach(btn => btn.addEventListener('click', async () => {
        if (!await showConfirm('Promote Version', 'Set this version as production?')) return;
        try { await api.admin.promoteModelVersion(parseInt(btn.dataset.id)); showToast('Promoted to production!', 'success'); loadModelVersions(versionsPage); loadModelRegistrySummary(); }
        catch (err) { showToast(err.message, 'error'); }
      }));
      tbody.querySelectorAll('.rollback-version-btn').forEach(btn => btn.addEventListener('click', async () => {
        try { await api.admin.setRollbackTarget(parseInt(btn.dataset.id)); showToast('Set as rollback target', 'warning'); loadModelVersions(versionsPage); loadModelRegistrySummary(); }
        catch (err) { showToast(err.message, 'error'); }
      }));
    }
    const totalPages = Math.ceil((data.total || 0) / versionsPerPage);
    renderPagination(document.getElementById('versionsPagination'), page, totalPages, loadModelVersions);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="table-empty">Error: ${esc(err.message)}</td></tr>`;
  }
}

async function loadModelRegistrySummary() {
  const container = document.getElementById('modelRegistrySummaryBody');
  if (!container) return;

  container.innerHTML = '<div class="table-empty"><div class="spinner"></div></div>';
  try {
    const data = await api.admin.getModelRegistrySummary();
    const items = data.items || [];
    if (!items.length) {
      container.innerHTML = '<div class="table-empty">No model registry data found.</div>';
      return;
    }

    container.innerHTML = items.map(item => {
      const current = item.current_production;
      const previous = item.previous;
      const rollback = item.rollback_target;
      return `
        <div class="model-summary-item">
          <div class="model-summary-title">${esc(item.model_name)}</div>
          <div class="model-summary-row"><span>Current:</span><strong>${esc(current?.version_tag || '—')}</strong></div>
          <div class="model-summary-row"><span>Previous:</span><strong>${esc(previous?.version_tag || '—')}</strong></div>
          <div class="model-summary-row"><span>Rollback:</span><strong>${esc(rollback?.version_tag || '—')}</strong></div>
          <div class="model-summary-row model-summary-muted"><span>Total Versions:</span><strong>${item.total_versions || 0}</strong></div>
        </div>
      `;
    }).join('');
  } catch (err) {
    container.innerHTML = `<div class="table-empty">Error: ${esc(err.message)}</div>`;
  }
}

function setupModelRegistryPanel() {
  document.getElementById('registerVersionBtn').addEventListener('click', () => {
    document.getElementById('registerVersionModal').classList.remove('hidden');
    document.getElementById('registerVersionFeedback').textContent = '';
  });
  document.getElementById('registerVersionModalClose').addEventListener('click', () => document.getElementById('registerVersionModal').classList.add('hidden'));
  document.getElementById('registerVersionCancel').addEventListener('click', () => document.getElementById('registerVersionModal').classList.add('hidden'));
  document.getElementById('registerVersionSave').addEventListener('click', async () => {
    const payload = {
      model_name: document.getElementById('regModelName').value.trim(),
      version_tag: document.getElementById('regVersionTag').value.trim(),
      artifact_uri: document.getElementById('regArtifactUri').value.trim(),
      training_job_id: document.getElementById('regTrainingJobId').value.trim(),
      notes: document.getElementById('regNotes').value.trim(),
    };
    const fb = document.getElementById('registerVersionFeedback');
    if (!payload.model_name || !payload.version_tag) { fb.textContent = 'Model name and version tag required.'; return; }
    try {
      fb.textContent = 'Registering...';
      await api.admin.createModelVersion(payload);
      showToast('Version registered!', 'success');
      document.getElementById('registerVersionModal').classList.add('hidden');
      loadModelVersions(versionsPage);
      loadModelRegistrySummary();
    } catch (err) { fb.textContent = err.message; }
  });
}

// =====================================================
// MCQ Observability Panel
// =====================================================
let mcqTimeChart = null;
let mcqModeChart = null;
let mcqObsAutoRefreshTimer = null;

async function loadMcqObservability() {
  const days = parseInt(document.getElementById('mcqObsDays').value || 30);
  const subject = document.getElementById('mcqObsSubject').value.trim();
  try {
    const data = await api.admin.getMcqObservability({ days, subject });
    const s = data.summary || {};
    document.getElementById('mcqTotal').textContent = s.total ?? '—';
    document.getElementById('mcqSuccessRate').textContent = (s.success_rate ?? 0) + '%';
    document.getElementById('mcqFailureRate').textContent = (s.failure_rate ?? 0) + '%';
    document.getElementById('mcqFallbackRate').textContent = (s.fallback_rate ?? 0) + '%';
    document.getElementById('mcqGenerationCounts').textContent = `${s.questions_generated_total ?? 0} / ${s.questions_requested_total ?? 0}`;
    document.getElementById('mcqAvgLatency').textContent = s.avg_latency_ms ?? '—';

    // Time series chart
    const ts = data.time_series || [];
    const tsLabels = ts.map(t => t.date);
    const tsSuccess = ts.map(t => t.success);
    const tsFailure = ts.map(t => t.failure);
    const tsFallback = ts.map(t => t.fallback);

    const timeCtx = document.getElementById('mcqTimeChart')?.getContext('2d');
    if (timeCtx) {
      if (mcqTimeChart) { mcqTimeChart.destroy(); mcqTimeChart = null; }
      mcqTimeChart = new Chart(timeCtx, {
        type: 'line',
        data: {
          labels: tsLabels,
          datasets: [
            { label: 'Success', data: tsSuccess, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: true, tension: 0.3 },
            { label: 'Failure', data: tsFailure, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: 0.3 },
            { label: 'Fallback', data: tsFallback, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', fill: true, tension: 0.3 },
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: '#94a3b8' } } },
          scales: {
            y: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { ticks: { color: '#6b7280', maxTicksLimit: 8 }, grid: { display: false } }
          }
        }
      });
    }

    // Mode chart (donut)
    const modes = data.by_mode || {};
    const modeCtx = document.getElementById('mcqModeChart')?.getContext('2d');
    if (modeCtx) {
      if (mcqModeChart) { mcqModeChart.destroy(); mcqModeChart = null; }
      const modeEntries = Object.entries(modes);
      if (!modeEntries.length) {
        mcqModeChart = new Chart(modeCtx, {
          type: 'doughnut',
          data: { labels: ['No data'], datasets: [{ data: [1], backgroundColor: ['#334155'], borderColor: '#1a2236', borderWidth: 2 }] },
          options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } } }
        });
        return;
      }
      const modeColors = ['#818cf8', '#34d399', '#fbbf24', '#f87171', '#22d3ee', '#f472b6'];
      mcqModeChart = new Chart(modeCtx, {
        type: 'doughnut',
        data: {
          labels: modeEntries.map(([label]) => label),
          datasets: [{
            data: modeEntries.map(([, value]) => value),
            backgroundColor: modeColors,
            borderColor: '#1a2236',
            borderWidth: 3,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: '#94a3b8' } } }
        }
      });
    }
  } catch (err) {
    showToast('Failed to load MCQ observability: ' + err.message, 'error');
  }
}

function setupMcqObsPanel() {
  document.getElementById('refreshMcqObsBtn').addEventListener('click', loadMcqObservability);
  document.getElementById('mcqObsDays').addEventListener('change', loadMcqObservability);
  document.getElementById('mcqObsSubject').addEventListener('keydown', e => { if (e.key === 'Enter') loadMcqObservability(); });
  if (mcqObsAutoRefreshTimer) clearInterval(mcqObsAutoRefreshTimer);
  mcqObsAutoRefreshTimer = setInterval(() => {
    const panel = document.getElementById('panel-mcq-obs');
    if (panel && panel.classList.contains('active')) {
      loadMcqObservability();
    }
  }, 15000);
}

// =====================================================
// Audit Trail Panel
// =====================================================
let auditPage = 1;
const auditPerPage = 15;

async function loadAuditLogs(page = 1) {
  auditPage = page;
  const params = {
    page, per_page: auditPerPage,
    action: document.getElementById('auditActionFilter').value.trim(),
    target_type: document.getElementById('auditTargetType').value,
    date_from: document.getElementById('auditDateFrom').value,
    date_to: document.getElementById('auditDateTo').value,
  };
  const tbody = document.getElementById('auditTableBody');
  tbody.innerHTML = `<tr><td colspan="7" class="table-empty"><div class="spinner"></div></td></tr>`;
  try {
    const data = await api.admin.listAuditLogs(params);
    const items = data.items || [];
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="table-empty">No audit logs found.</td></tr>`;
    } else {
      tbody.innerHTML = items.map((log, index) => `<tr>
        <td>${((page - 1) * auditPerPage) + index + 1}</td>
        <td><span class="badge badge-purple">${esc(log.action)}</span></td>
        <td>${esc(log.actor_name || (log.actor_id ? `ID:${log.actor_id}` : '—'))}</td>
        <td>${log.target_type ? `<span class="badge badge-gray">${esc(log.target_type)}</span> ${esc(log.target_id || '')}` : '—'}</td>
        <td style="color:#6b7280;font-size:0.75rem">${esc(log.ip || '—')}</td>
        <td style="color:#6b7280;font-size:0.78rem;max-width:200px;overflow:hidden;text-overflow:ellipsis">${esc(log.notes || '—')}</td>
        <td style="font-size:0.78rem;color:#6b7280">${log.created_at ? new Date(log.created_at).toLocaleString() : '—'}</td>
      </tr>`).join('');
    }
    const totalPages = Math.ceil((data.total || 0) / auditPerPage);
    renderPagination(document.getElementById('auditPagination'), page, totalPages, loadAuditLogs);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="table-empty">Error: ${esc(err.message)}</td></tr>`;
  }
}

function setupAuditPanel() {
  document.getElementById('auditSearchBtn').addEventListener('click', () => loadAuditLogs(1));
  ['auditActionFilter', 'auditTargetType', 'auditDateFrom', 'auditDateTo'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => loadAuditLogs(1));
  });
  document.getElementById('exportAuditBtn').addEventListener('click', async () => {
    try {
      const csv = await api.admin.exportAuditLogs({
        action: document.getElementById('auditActionFilter').value.trim(),
        target_type: document.getElementById('auditTargetType').value,
        date_from: document.getElementById('auditDateFrom').value,
        date_to: document.getElementById('auditDateTo').value,
      });
      downloadText(csv, `audit_logs_${today()}.csv`, 'text/csv');
      showToast('Audit CSV downloaded', 'success');
    } catch (err) { showToast(err.message, 'error'); }
  });
}

// =====================================================
// Utilities
// =====================================================
function esc(str) {
  return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

function downloadText(text, filename, mime) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// =====================================================
// Main Init
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
  const session = enforceAdmin();
  if (!session) return;

  const user = session.user;

  // Setup navbar / sidebar user info
  ['sidebarUserName', 'topbarUserName'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = user.name || 'Admin';
  });
  ['sidebarAvatar', 'topbarAvatar'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = (user.name || 'A').charAt(0).toUpperCase();
  });

  // Profile Dropdown Toggle Logic
  const profileBtn = document.getElementById('profile-btn');
  const dropdown = document.getElementById('profile-dropdown');
  const profileMenuContainer = document.getElementById('profile-menu-container');

  if (profileBtn && dropdown && profileMenuContainer) {
    profileBtn.addEventListener('click', (e) => {
      e.stopPropagation(); // Prevent immediate closing
      dropdown.classList.toggle('hidden');
    });

    document.addEventListener('click', (event) => {
      if (!profileMenuContainer.contains(event.target)) {
        dropdown.classList.add('hidden');
      }
    });
  }

  // Logout (Sidebar & Topbar) with Global Confirmation Dialog
  const performLogout = async () => {
    const isConfirmed = await showConfirm('Log Out', 'Are you sure you want to log out?');
    if (isConfirmed) {
      clearSession();
      window.location.replace('/index.html');
    } else {
      // Close dropdown if user cancels
      if (dropdown) dropdown.classList.add('hidden');
    }
  };

  document.getElementById('sidebarLogoutBtn')?.addEventListener('click', performLogout);
  document.getElementById('topbarLogoutBtn')?.addEventListener('click', performLogout);

  // Mobile sidebar toggle
  document.getElementById('sidebarToggle')?.addEventListener('click', () => {
    document.getElementById('adminSidebar').classList.toggle('open');
  });

  // Health retry
  document.getElementById('healthRetryBtn')?.addEventListener('click', checkServer);

  // Panel navigation
  setupPanelNav();
  setupQuickActions();

  // Setup all panels
  setupUsersPanel();
  // setupSchoolsPanel();
  setupTestResultsPanel();
  setupTrainingPanel();
  setupModelRegistryPanel();
  setupMcqObsPanel();
  setupAuditPanel();

  // Dashboard refresh
  document.getElementById('refreshStatsBtn')?.addEventListener('click', () => loadPanelData('dashboard', true));

  // Initial loads
  checkServer();
  activatePanel('dashboard', true);
  loadSchools(); // Also populates allSchools for user filters
  // Keep pending teacher requests badge up to date.
});
