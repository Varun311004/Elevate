// Application State Management
export const state = {
  // User Session
  currentUser: null,
  currentGrade: 'elementary',
  
  // Quiz/Learning State
  currentQuestion: null,
  questionsAnswered: 0,
  correctAnswers: 0,
  currentEmotion: 'neutral',
  sessionStartTime: null,
  
  // Camera & Detection State
  cameraStream: null,
  cameraActive: false,
  faceDetectionConfirmed: false,
  emotionDetectionInterval: null,
  modelsLoaded: false,
  cameraPermissionDenied: false,
  usingSimulatedEmotions: false,
  modelLoadRetries: 0,
  maxModelRetries: 3,
  
  // Emotion Tracking
  emotionHistory: [],
  emotionCounts: {
    happy: 0,
    bored: 0,
    focused: 0,
    confused: 0,
    neutral: 0,
    angry: 0,
    surprised: 0
  },
  
  // Timers
  breakTimer: null,
  breakTimeRemaining: 120,
  
  // Scoring
  subjectScores: {
    Mathematics: { correct: 0, total: 0, previousScore: 0, previousTotal: 0 },
    Science: { correct: 0, total: 0, previousScore: 0, previousTotal: 0 },
    Technology: { correct: 0, total: 0, previousScore: 0, previousTotal: 0 },
    Engineering: { correct: 0, total: 0, previousScore: 0, previousTotal: 0 }
  },
  
  // Question Management
  usedQuestions: new Set(),
  testQuestions: [],
  currentTestQuestionIndex: 0,
  isTestMode: false,
  selectedSubject: null,
  selectedPracticeDifficulty: 'adaptive',
  questions: [], // Current loaded questions from API
  currentQuestionIndex: 0,
  questionStartTime: null,
  
  // Settings & Profile
  debugMode: false,
  preferredSubjects: [],
  userProfile: {
    name: '',
    email: '',
    grade: 'elementary',
    totalQuestions: 0,
    totalCorrect: 0,
    streak: 0,
    level: 'Beginner',
    joinDate: new Date().toISOString(),
    lastLoginDate: null
  }
};

// State getter/setter functions
export function getState() {
  return state;
}

export function updateState(updates) {
  Object.assign(state, updates);
}

export function resetState() {
  state.questionsAnswered = 0;
  state.correctAnswers = 0;
  state.emotionHistory = [];
  state.usedQuestions.clear();
  state.testQuestions = [];
  state.currentTestQuestionIndex = 0;
  state.sessionStartTime = null;
  state.cameraStream = null;
  state.cameraActive = false;
  state.emotionCounts = {
    happy: 0,
    bored: 0,
    focused: 0,
    confused: 0,
    neutral: 0,
    angry: 0,
    surprised: 0
  };
}