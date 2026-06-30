// Utility functions
export const utils = {
  _notificationTimer: null,

  messageCatalog: {
    'generic.completed': 'Done successfully.',
    'generic.failed': 'Something went wrong. Please try again.',
    'generic.network_failed': 'We could not connect right now. Please check your connection and try again.',
    'auth.login_success': 'Signed in successfully. Redirecting to your dashboard.',
    'auth.login_failed': 'We could not sign you in. Please check your details and try again.',
    'auth.signup_success': 'Your account is ready. Redirecting to your dashboard.',
    'auth.signup_failed': 'We could not create your account right now. Please try again.',
    'teacher.test_create_success': 'Test created successfully.',
    'teacher.test_create_failed': 'We could not create the test right now. Please try again.',
    'teacher.test_update_success': 'Test updated successfully.',
    'teacher.test_update_failed': 'We could not update the test right now. Please try again.',
    'teacher.test_delete_success': 'Test deleted successfully.',
    'teacher.test_delete_failed': 'We could not delete the test right now. Please try again.',
    'teacher.classroom_create_success': 'Classroom created successfully.',
    'teacher.classroom_create_failed': 'We could not create the classroom right now. Please try again.',
    'teacher.assignment_create_success': 'Assignment created successfully.',
    'teacher.assignment_create_failed': 'We could not create the assignment right now. Please try again.',
    'teacher.student_add_success': 'Student added successfully.',
    'teacher.student_add_failed': 'We could not add the student right now. Please try again.',
    'teacher.student_remove_success': 'Student removed successfully.',
    'teacher.student_remove_failed': 'We could not remove the student right now. Please try again.',
  },

  getMessage(messageOrKey, replacements = {}) {
    const raw = this.messageCatalog[messageOrKey] || messageOrKey || this.messageCatalog['generic.failed'];
    return String(raw).replace(/\{([^}]+)\}/g, (_, token) => {
      const value = replacements[token];
      return value === undefined || value === null ? '' : String(value);
    });
  },

  notifySuccess(messageOrKey, replacements = {}, options = {}) {
    const message = this.getMessage(messageOrKey, replacements);
    this.showNotification(message, 'success', options);
  },

  notifyError(messageOrKey, replacements = {}, options = {}) {
    const message = this.getMessage(messageOrKey, replacements);
    this.showNotification(message, 'error', options);
  },

  ensureNotificationElements() {
    let feedback = document.getElementById('emotionFeedback');
    let icon = document.getElementById('feedbackIcon');
    let text = document.getElementById('feedbackText');

    // Support legacy container id used by some pages.
    if (!feedback) {
      const legacy = document.getElementById('emotion-feedback');
      if (legacy) {
        legacy.id = 'emotionFeedback';
        feedback = legacy;
      }
    }

    if (!feedback) {
      feedback = document.createElement('div');
      feedback.id = 'emotionFeedback';
      feedback.className = 'emotion-feedback';
      feedback.innerHTML = `
        <span class="emotion-feedback-icon" id="feedbackIcon">ℹ️</span>
        <div class="emotion-feedback-text" id="feedbackText">Notification</div>
      `;
      document.body.appendChild(feedback);
    }

    icon = document.getElementById('feedbackIcon');
    text = document.getElementById('feedbackText');

    if (!icon) {
      icon = document.createElement('span');
      icon.id = 'feedbackIcon';
      icon.className = 'emotion-feedback-icon';
      icon.textContent = 'ℹ️';
      feedback.prepend(icon);
    }

    if (!text) {
      text = document.createElement('div');
      text.id = 'feedbackText';
      text.className = 'emotion-feedback-text';
      text.textContent = 'Notification';
      feedback.appendChild(text);
    }

    return { feedback, icon, text };
  },

  // Show notification/feedback (Consolidates showEmotionFeedback from index5.html)
  showNotification(message, type = 'info', options = {}) {
    // Check if notifications are enabled (import storage dynamically to avoid circular deps)
    try {
      const savedSettings = localStorage.getItem('userSettings');
      if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        if (settings.enableNotifications === false) {
          return; // Don't show notification if disabled
        }
      }
    } catch (e) {
      // If settings check fails, show notification anyway
    }
    
    const { feedback, icon, text } = this.ensureNotificationElements();
    const resolvedMessage = this.getMessage(message);
    const baseDuration = Number(options?.durationMs || options?.duration || 0);
    const adaptiveDuration = Math.min(6500, Math.max(2600, resolvedMessage.length * 40));
    const durationMs = Number.isFinite(baseDuration) && baseDuration > 0 ? baseDuration : adaptiveDuration;

    // Icons matching index5.html exactly + extras
    const icons = {
      positive: '😊',
      encouragement: '💪',
      error: '❌',
      info: 'ℹ️',
      // Fallbacks
      success: '✅',
      warning: '⚠️',
      confused: '😕',
      focused: '🧠',
      bored: '😑',
      angry: '😠',
      surprised: '😮'
    };

    icon.textContent = icons[type] || icons.info;
    text.textContent = resolvedMessage;
    
    feedback.classList.add('show');

    // Hide after duration, resetting previous timers to keep timing accurate.
    if (this._notificationTimer) {
      clearTimeout(this._notificationTimer);
      this._notificationTimer = null;
    }
    this._notificationTimer = setTimeout(() => {
      feedback.classList.remove('show');
      this._notificationTimer = null;
    }, durationMs);
  },

  // Format time (seconds to MM:SS)
  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  },

  // Calculate accuracy percentage
  calculateAccuracy(correct, total) {
    if (total === 0) return 0;
    return Math.round((correct / total) * 100);
  },

  // Grade hierarchy and management
  gradeHierarchy: ['elementary', 'middle', 'high', 'college'],
  
  // Get accessible grades based on user's grade (hierarchical access)
  // Elementary can access ALL levels, Middle can access Middle+High+College, etc.
  getAccessibleGrades(userGrade) {
    if (!userGrade) return this.gradeHierarchy;
    
    const userIndex = this.gradeHierarchy.indexOf(userGrade);
    if (userIndex === -1) return this.gradeHierarchy;
    
    // User can access their grade and all HIGHER grades
    // Elementary (index 0) → slice(0) → all grades
    // Middle (index 1) → slice(1) → middle, high, college
    // High (index 2) → slice(2) → high, college
    // College (index 3) → slice(3) → college only
    return this.gradeHierarchy.slice(userIndex);
  },
  
  // Get grade display name
  getGradeDisplayName(grade) {
    const gradeNames = {
      'elementary': 'Elementary (K-5)',
      'middle': 'Middle School (6-8)',
      'high': 'High School (9-12)',
      'college': 'College'
    };
    return gradeNames[grade] || grade;
  },

  // Get random element from array
  randomElement(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  },

  // Shuffle array
  shuffleArray(arr) {
    const newArr = [...arr];
    for (let i = newArr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [newArr[i], newArr[j]] = [newArr[j], newArr[i]];
    }
    return newArr;
  },

  // Calculate Streak (from index5.html logic)
  calculateStreak(lastLoginDate) {
    const today = new Date().toDateString();
    
    if (!lastLoginDate || lastLoginDate !== today) {
        return 1;
    } else {
        // Logic for consecutive days could go here, simplistic for now matching source
        const daysSinceLastLogin = Math.floor((Date.now() - new Date(lastLoginDate).getTime()) / (1000 * 60 * 60 * 24));
        if (daysSinceLastLogin > 1) {
            return 0;
        }
        return 1; // Maintained if logged in today
    }
  },

  // Generate unique ID
  generateId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  },

  // Safe DOM query
  $(selector) {
    return document.querySelector(selector);
  },

  $$(selector) {
    return document.querySelectorAll(selector);
  },

  // Add event listener with cleanup
  addListener(element, event, handler) {
    if (element) {
      element.addEventListener(event, handler);
      return () => element.removeEventListener(event, handler);
    }
    return () => {};
  },

  // Navigate to page - full page load with optional history replace
  navigateTo(page, useReplace = false) {
    // Mark navigation in progress to handle any cleanup
    sessionStorage.setItem('navigating', 'true');
    
    if (useReplace) {
      // Replace current history entry (avoids stacking auth pages)
      window.location.replace(page);
    } else {
      // Standard navigation - keeps entry in back stack
      window.location.href = page;
    }

    // TODO Phase 8: Convert to SPA with History API routing for smoother transitions
  },

  // Get URL parameter
  getUrlParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
  }
};