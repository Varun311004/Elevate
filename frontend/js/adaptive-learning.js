// Adaptive Learning Algorithm
// Adjusts difficulty based on user performance and emotional state
import { state, updateState } from './state.js';
import { config } from './config.js';

// Difficulty adjustment settings
const DIFFICULTY_SETTINGS = {
  // Minimum questions before difficulty adjustment
  MIN_QUESTIONS_FOR_ADJUSTMENT: 3,
  
  // Accuracy thresholds for difficulty changes
  INCREASE_THRESHOLD: 0.80, // 80% correct
  DECREASE_THRESHOLD: 0.50, // 50% correct
  
  // Emotion-based modifiers
  EMOTION_MODIFIERS: {
    happy: 0.05,
    bored: 0.10,
    focused: 0.15,
    confused: -0.20,
    neutral: 0,
    angry: -0.25,
    surprised: 0.06,
  },
  
  // Cooldown period (number of questions) before next adjustment
  ADJUSTMENT_COOLDOWN: 5,
  
  // Maximum difficulty changes per session
  MAX_CHANGES_PER_SESSION: 10
};

// Difficulty levels
const DIFFICULTY_LEVELS = ['easy', 'medium', 'hard', 'expert'];

// Track adjustment history
const adjustmentHistory = {
  lastAdjustmentQuestion: 0,
  totalAdjustments: 0,
  subjectDifficulty: {} // Per-subject difficulty tracking
};

// Initialize adaptive learning for a subject
export function initializeSubject(subject) {
  if (!adjustmentHistory.subjectDifficulty[subject]) {
    console.log(`🎯 Initializing adaptive learning for: ${subject}`);
    adjustmentHistory.subjectDifficulty[subject] = {
      currentDifficulty: 'medium',
      difficultyIndex: 1, // 0=easy, 1=medium, 2=hard, 3=expert
      questionCount: 0,
      correctCount: 0,
      emotionSamples: [],
      currentStreak: 0,
      bestStreak: 0,
      avgResponseTime: 0,
      totalResponseTime: 0,
      answeredQuestionIds: new Set() // Track answered questions to prevent repetition
    };
  }
}

// Get current difficulty for a subject
export function getCurrentDifficulty(subject) {
  initializeSubject(subject);
  return adjustmentHistory.subjectDifficulty[subject].currentDifficulty;
}

// Record answer result
export function recordAnswer(subject, isCorrect, emotion = null, responseTime = 0, questionId = null) {
  initializeSubject(subject);
  
  const subjectData = adjustmentHistory.subjectDifficulty[subject];
  subjectData.questionCount++;
  
  // Track answered questions to prevent repetition
  if (questionId) {
    subjectData.answeredQuestionIds.add(questionId);
  }
  
  // Update correctness and streaks
  if (isCorrect) {
    subjectData.correctCount++;
    subjectData.currentStreak++;
    if (subjectData.currentStreak > subjectData.bestStreak) {
      subjectData.bestStreak = subjectData.currentStreak;
    }
  } else {
    subjectData.currentStreak = 0; // Reset streak on wrong answer
  }
  
  // Track response time
  if (responseTime > 0) {
    subjectData.totalResponseTime += responseTime;
    subjectData.avgResponseTime = subjectData.totalResponseTime / subjectData.questionCount;
  }
  
  // Store emotion sample if provided
  if (emotion) {
    subjectData.emotionSamples.push(emotion);
    // Keep only last 10 emotion samples
    if (subjectData.emotionSamples.length > 10) {
      subjectData.emotionSamples.shift();
    }
  }
  
  // Check if adjustment is needed
  if (shouldAdjustDifficulty(subjectData)) {
    const adjustmentResult = adjustDifficulty(subject);
    return adjustmentResult; // Return adjustment info for notifications
  }
  
  return { adjusted: false };
}

// Determine if difficulty should be adjusted
function shouldAdjustDifficulty(subjectData) {
  // Need minimum questions
  if (subjectData.questionCount < DIFFICULTY_SETTINGS.MIN_QUESTIONS_FOR_ADJUSTMENT) {
    return false;
  }
  
  // Check cooldown
  const questionsSinceLastAdjustment = 
    state.questionsAnswered - adjustmentHistory.lastAdjustmentQuestion;
  if (questionsSinceLastAdjustment < DIFFICULTY_SETTINGS.ADJUSTMENT_COOLDOWN) {
    return false;
  }
  
  // Check max adjustments per session
  if (adjustmentHistory.totalAdjustments >= DIFFICULTY_SETTINGS.MAX_CHANGES_PER_SESSION) {
    return false;
  }
  
  return true;
}

// Adjust difficulty based on performance and emotion
function adjustDifficulty(subject) {
  const subjectData = adjustmentHistory.subjectDifficulty[subject];
  
  // Calculate accuracy
  const accuracy = subjectData.correctCount / subjectData.questionCount;
  
  // Calculate emotion modifier (average of recent emotions)
  let emotionModifier = 0;
  if (subjectData.emotionSamples.length > 0) {
    const emotionSum = subjectData.emotionSamples.reduce((sum, emotion) => {
      return sum + (DIFFICULTY_SETTINGS.EMOTION_MODIFIERS[emotion] || 0);
    }, 0);
    emotionModifier = emotionSum / subjectData.emotionSamples.length;
  }
  
  // Determine adjustment direction
  let adjustment = 0;
  
  if (accuracy >= DIFFICULTY_SETTINGS.INCREASE_THRESHOLD) {
    adjustment = 1; // Increase difficulty
  } else if (accuracy <= DIFFICULTY_SETTINGS.DECREASE_THRESHOLD) {
    adjustment = -1; // Decrease difficulty
  }
  
  // Apply emotion modifier
  if (emotionModifier > 0.1 && accuracy > 0.6) {
    adjustment = Math.max(adjustment, 1); // User is engaged, increase
  } else if (emotionModifier < -0.15) {
    adjustment = Math.min(adjustment, -1); // User is struggling, decrease
  }
  
  // Apply adjustment
  if (adjustment !== 0) {
    const newIndex = Math.max(0, Math.min(
      DIFFICULTY_LEVELS.length - 1,
      subjectData.difficultyIndex + adjustment
    ));
    
    if (newIndex !== subjectData.difficultyIndex) {
      const oldDifficulty = subjectData.currentDifficulty;
      subjectData.difficultyIndex = newIndex;
      subjectData.currentDifficulty = DIFFICULTY_LEVELS[newIndex];
      
      // Record adjustment
      adjustmentHistory.lastAdjustmentQuestion = state.questionsAnswered;
      adjustmentHistory.totalAdjustments++;
      
      // Reset counters for next adjustment period
      subjectData.questionCount = 0;
      subjectData.correctCount = 0;
      
      console.log(`Difficulty adjusted for ${subject}: ${oldDifficulty} → ${subjectData.currentDifficulty}`);
      
      return {
        adjusted: true,
        oldDifficulty,
        newDifficulty: subjectData.currentDifficulty,
        reason: adjustment > 0 ? 'performance_high' : 'performance_low',
        emotionInfluence: emotionModifier
      };
    }
  }
  
  return { adjusted: false };
}

// Get recommended difficulty for next question
export function getRecommendedDifficulty(subject) {
  initializeSubject(subject);
  return adjustmentHistory.subjectDifficulty[subject].currentDifficulty;
}

// Get performance summary for a subject
export function getSubjectPerformance(subject) {
  if (!adjustmentHistory.subjectDifficulty[subject]) {
    return null;
  }
  
  const data = adjustmentHistory.subjectDifficulty[subject];
  const accuracy = data.questionCount > 0 
    ? (data.correctCount / data.questionCount) * 100 
    : 0;
  
  return {
    subject,
    difficulty: data.currentDifficulty,
    questionsAnswered: data.questionCount,
    accuracy: accuracy.toFixed(1),
    currentStreak: data.currentStreak || 0,
    bestStreak: data.bestStreak || 0,
    avgResponseTime: data.avgResponseTime ? Math.round(data.avgResponseTime) : 0,
    recentEmotions: data.emotionSamples.slice(-5)
  };
}

// Get all subjects performance
export function getAllPerformance() {
  return Object.keys(adjustmentHistory.subjectDifficulty).map(subject => 
    getSubjectPerformance(subject)
  );
}

// Reset adaptive learning data
export function resetAdaptiveLearning() {
  adjustmentHistory.lastAdjustmentQuestion = 0;
  adjustmentHistory.totalAdjustments = 0;
  adjustmentHistory.subjectDifficulty = {};
}

// Check if a question was already answered for a subject
export function isQuestionAnswered(subject, questionId) {
  if (!adjustmentHistory.subjectDifficulty[subject]) {
    return false;
  }
  return adjustmentHistory.subjectDifficulty[subject].answeredQuestionIds.has(questionId);
}

// Get list of answered question IDs for a subject
export function getAnsweredQuestionIds(subject) {
  if (!adjustmentHistory.subjectDifficulty[subject]) {
    return [];
  }
  return Array.from(adjustmentHistory.subjectDifficulty[subject].answeredQuestionIds);
}

// Export adaptive learning system
export const adaptiveLearning = {
  initializeSubject,
  getCurrentDifficulty,
  recordAnswer,
  getRecommendedDifficulty,
  getSubjectPerformance,
  getAllPerformance,
  resetAdaptiveLearning,
  isQuestionAnswered,
  getAnsweredQuestionIds,
  DIFFICULTY_LEVELS
};
