// Progress Tracker
// Handles progress display, tracking, and persistence
import { state, updateState } from './state.js';
import { storage } from './storage.js';
import { utils } from './utils.js';
import { config } from './config.js';

// Progress tracking settings
const PROGRESS_SETTINGS = {
  // Auto-save interval (milliseconds)
  AUTO_SAVE_INTERVAL: 30000, // 30 seconds
  
  // Level progression
  QUESTIONS_PER_LEVEL: 50,
  
  // Streak settings
  MAX_STREAK_BONUS: 10,
  STREAK_MILESTONE: 7 // Days
};

// Auto-save timer
let autoSaveTimer = null;

// Start auto-save
export function startAutoSave() {
  if (autoSaveTimer) {
    clearInterval(autoSaveTimer);
  }
  
  autoSaveTimer = setInterval(() => {
    saveProgress();
  }, PROGRESS_SETTINGS.AUTO_SAVE_INTERVAL);
}

// Stop auto-save
export function stopAutoSave() {
  if (autoSaveTimer) {
    clearInterval(autoSaveTimer);
    autoSaveTimer = null;
  }
}

// Save progress to storage
export function saveProgress() {
  const progress = {
    userProfile: state.userProfile,
    questionsAnswered: state.questionsAnswered,
    correctAnswers: state.correctAnswers,
    subjectScores: state.subjectScores,
    emotionCounts: state.emotionCounts,
    lastUpdated: new Date().toISOString()
  };
  
  storage.saveProgress(progress);
  console.log('Progress saved');
}

// Load progress from storage
export function loadProgress() {
  const progress = storage.loadProgress();
  
  if (progress) {
    updateState({
      userProfile: progress.userProfile,
      questionsAnswered: progress.questionsAnswered || 0,
      correctAnswers: progress.correctAnswers || 0,
      subjectScores: progress.subjectScores || {},
      emotionCounts: progress.emotionCounts || {}
    });
    
    console.log('Progress loaded');
    return true;
  }
  
  return false;
}

// Update question progress
export function updateQuestionProgress(subject, isCorrect) {
  // Update global counters
  updateState({
    questionsAnswered: state.questionsAnswered + 1,
    correctAnswers: state.correctAnswers + (isCorrect ? 1 : 0)
  });
  
  // Update subject scores
  if (!state.subjectScores[subject]) {
    state.subjectScores[subject] = {
      correct: 0,
      total: 0,
      previousScore: 0
    };
  }
  
  state.subjectScores[subject].total++;
  if (isCorrect) {
    state.subjectScores[subject].correct++;
  }
  
  // Update user profile
  updateState({
    userProfile: {
      ...state.userProfile,
      totalQuestions: state.questionsAnswered,
      totalCorrect: state.correctAnswers,
      level: calculateLevel(state.questionsAnswered)
    }
  });
  
  // Update UI
  updateProgressDisplay();
}

// Calculate level based on questions answered
function calculateLevel(questionsAnswered) {
  return Math.floor(questionsAnswered / PROGRESS_SETTINGS.QUESTIONS_PER_LEVEL) + 1;
}

// Update progress display on UI
export function updateProgressDisplay() {
  // Dashboard stats
  updateDashboardStats();
  
  // Subject performance
  updateSubjectPerformance();
  
  // Streak display
  updateStreakDisplay();
}

// Update dashboard statistics
function updateDashboardStats() {
  const totalQuestions = document.getElementById('dashboardTotalQuestions');
  const accuracy = document.getElementById('dashboardAccuracy');
  const streak = document.getElementById('dashboardStreak');
  const level = document.getElementById('dashboardLevel');
  
  if (totalQuestions) {
    totalQuestions.textContent = state.userProfile.totalQuestions;
  }
  
  if (accuracy) {
    const acc = utils.calculateAccuracy(
      state.userProfile.totalCorrect,
      state.userProfile.totalQuestions
    );
    accuracy.textContent = acc + '%';
  }
  
  if (streak) {
    streak.textContent = state.userProfile.streak;
  }
  
  if (level) {
    level.textContent = state.userProfile.level;
  }
}

// Update subject performance display
function updateSubjectPerformance() {
  const container = document.getElementById('subjectPerformanceGrid');
  if (!container) return;
  
  container.innerHTML = '';
  
  Object.keys(state.subjectScores).forEach(subject => {
    const score = state.subjectScores[subject];
    const accuracy = score.total > 0 
      ? Math.round((score.correct / score.total) * 100) 
      : 0;
    
    const change = accuracy - score.previousScore;
    let trendClass = 'stable';
    let trendIcon = 'fa-minus';
    
    if (change > 5) {
      trendClass = 'improving';
      trendIcon = 'fa-arrow-up';
    } else if (change < -5) {
      trendClass = 'declining';
      trendIcon = 'fa-arrow-down';
    }
    
    const subjectConfig = config.SUBJECTS[subject] || { icon: 'fa-book', color: '#6c757d' };
    
    const item = document.createElement('div');
    item.className = `subject-performance-item ${trendClass}`;
    item.innerHTML = `
      <div class="subject-icon" style="background: ${subjectConfig.color}">
        <i class="fas ${subjectConfig.icon}"></i>
      </div>
      <div class="subject-info">
        <div class="subject-name">${subject}</div>
        <div class="subject-stats">
          <span>${score.correct}/${score.total} correct</span>
          <span class="accuracy">${accuracy}%</span>
        </div>
      </div>
      <div class="subject-trend">
        <i class="fas ${trendIcon}"></i>
      </div>
    `;
    
    container.appendChild(item);
  });
}

// Update streak display
function updateStreakDisplay() {
  const streakElement = document.getElementById('currentStreak');
  if (!streakElement) return;
  
  const streak = state.userProfile.streak || 0;
  streakElement.textContent = `${streak} day${streak !== 1 ? 's' : ''}`;
  
  // Show milestone notification
  if (streak > 0 && streak % PROGRESS_SETTINGS.STREAK_MILESTONE === 0) {
    utils.showNotification(
      `Amazing! ${streak} day streak! Keep it up! 🔥`,
      'success'
    );
  }
}

// Get progress summary
export function getProgressSummary() {
  const totalQuestions = state.userProfile.totalQuestions;
  const totalCorrect = state.userProfile.totalCorrect;
  const accuracy = utils.calculateAccuracy(totalCorrect, totalQuestions);
  
  return {
    totalQuestions,
    totalCorrect,
    accuracy,
    level: state.userProfile.level,
    streak: state.userProfile.streak,
    questionsToNextLevel: PROGRESS_SETTINGS.QUESTIONS_PER_LEVEL - 
      (totalQuestions % PROGRESS_SETTINGS.QUESTIONS_PER_LEVEL),
    subjectBreakdown: Object.keys(state.subjectScores).map(subject => {
      const score = state.subjectScores[subject];
      return {
        subject,
        correct: score.correct,
        total: score.total,
        accuracy: score.total > 0 
          ? Math.round((score.correct / score.total) * 100) 
          : 0
      };
    })
  };
}

// Export progress summary as JSON (for download/export)
export function exportProgress() {
  const summary = getProgressSummary();
  const exportData = {
    ...summary,
    emotionData: state.emotionCounts,
    exportDate: new Date().toISOString(),
    userProfile: state.userProfile
  };
  
  return JSON.stringify(exportData, null, 2);
}

// Check for achievements
export function checkAchievements() {
  const achievements = [];
  
  // First question
  if (state.questionsAnswered === 1) {
    achievements.push({
      title: 'First Step',
      description: 'Answered your first question!',
      icon: '🎯'
    });
  }
  
  // Perfect accuracy on 10 questions
  if (state.questionsAnswered >= 10 && 
      state.correctAnswers === state.questionsAnswered) {
    achievements.push({
      title: 'Perfect Start',
      description: '100% accuracy on first 10 questions!',
      icon: '💯'
    });
  }
  
  // Level milestones
  if (state.userProfile.level > 1 && 
      state.questionsAnswered % PROGRESS_SETTINGS.QUESTIONS_PER_LEVEL === 0) {
    achievements.push({
      title: `Level ${state.userProfile.level}`,
      description: 'Reached a new level!',
      icon: '⭐'
    });
  }
  
  // Streak milestones
  if (state.userProfile.streak > 0 && 
      state.userProfile.streak % PROGRESS_SETTINGS.STREAK_MILESTONE === 0) {
    achievements.push({
      title: `${state.userProfile.streak} Day Streak`,
      description: 'Consistent learning pays off!',
      icon: '🔥'
    });
  }
  
  // Show achievements
  achievements.forEach(achievement => {
    utils.showNotification(
      `${achievement.icon} ${achievement.title}: ${achievement.description}`,
      'success'
    );
  });
  
  return achievements;
}

// Export progress tracker
export const progressTracker = {
  startAutoSave,
  stopAutoSave,
  saveProgress,
  loadProgress,
  updateQuestionProgress,
  updateProgressDisplay,
  getProgressSummary,
  exportProgress,
  checkAchievements
};
