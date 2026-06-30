// Question Manager module
import { questionDatabase } from './question-database.js';
import { state, updateState } from './state.js';
import { utils } from './utils.js';
import { adaptiveLearning } from './adaptive-learning.js';
import { progressTracker } from './progress-tracker.js';

export const questionManager = {
  // Get questions for current grade and subject
  getQuestions(grade = state.currentGrade, subject = null) {
    const questions = questionDatabase[grade] || [];
    
    if (subject) {
      return questions.filter(q => q.category === subject);
    }
    
    return questions;
  },

  // Get a random unused question
  getNextQuestion(subject = state.selectedSubject) {
    const questions = this.getQuestions(state.currentGrade, subject);
    const unused = questions.filter(q => !state.usedQuestions.has(q.id));
    
    if (unused.length === 0) {
      // Reset if all questions used
      state.usedQuestions.clear();
      return utils.randomElement(questions);
    }
    
    const question = utils.randomElement(unused);
    state.usedQuestions.add(question.id);
    
    return question;
  },

  // Load and display question
  displayQuestion(question) {
    if (!question) return;
    
    updateState({ currentQuestion: question });
    
    // Update UI
    const categoryEl = document.getElementById('questionCategory');
    const difficultyEl = document.getElementById('difficultyBadge');
    const questionTextEl = document.getElementById('questionText');
    const optionsContainer = document.getElementById('answerOptions');
    
    if (categoryEl) categoryEl.textContent = question.category;
    
    if (difficultyEl) {
      difficultyEl.textContent = question.difficulty.charAt(0).toUpperCase() + question.difficulty.slice(1);
      difficultyEl.className = `difficulty-badge difficulty-${question.difficulty}`;
    }
    
    if (questionTextEl) questionTextEl.textContent = question.question;
    
    if (optionsContainer) {
      optionsContainer.innerHTML = question.options.map((option, index) => `
        <div class="answer-option" data-index="${index}">
          <span class="option-letter">${String.fromCharCode(65 + index)}.</span>
          <span class="option-text">${option}</span>
        </div>
      `).join('');
      
      // Add click handlers
      optionsContainer.querySelectorAll('.answer-option').forEach(opt => {
        opt.addEventListener('click', () => this.selectOption(opt));
      });
    }
    
    // Show/hide buttons
    const submitBtn = document.getElementById('submitAnswer');
    const nextBtn = document.getElementById('nextQuestion');
    const feedbackEl = document.getElementById('feedbackMessage');
    
    if (submitBtn) submitBtn.style.display = 'inline-block';
    if (nextBtn) nextBtn.style.display = 'none';
    if (feedbackEl) {
      feedbackEl.style.display = 'none';
      feedbackEl.className = 'feedback-message';
    }
  },

  // Select an option
  selectOption(element) {
    document.querySelectorAll('.answer-option').forEach(opt => {
      opt.classList.remove('selected');
    });
    element.classList.add('selected');
  },

  // Check answer
  checkAnswer() {
    const selected = document.querySelector('.answer-option.selected');
    if (!selected) {
      utils.showNotification('Please select an answer', 'warning');
      return;
    }
    
    const selectedIndex = parseInt(selected.dataset.index);
    const correct = selectedIndex === state.currentQuestion.correct;
    const subject = state.currentQuestion.category;
    
    // Update progress tracker
    progressTracker.updateQuestionProgress(subject, correct);
    
    // Record for adaptive learning (includes current emotion)
    const currentEmotion = state.currentEmotion || 'neutral';
    adaptiveLearning.recordAnswer(subject, correct, currentEmotion);
    
    // Show feedback
    this.showFeedback(correct);
    
    // Visual feedback on options
    document.querySelectorAll('.answer-option').forEach((opt, idx) => {
      if (idx === state.currentQuestion.correct) {
        opt.classList.add('correct');
      } else if (idx === selectedIndex && !correct) {
        opt.classList.add('incorrect');
      }
    });
    
    // Hide submit, show next
    const submitBtn = document.getElementById('submitAnswer');
    const nextBtn = document.getElementById('nextQuestion');
    if (submitBtn) submitBtn.style.display = 'none';
    if (nextBtn) nextBtn.style.display = 'inline-block';
    
    return correct;
  },

  // Show feedback
  showFeedback(correct) {
    const feedbackEl = document.getElementById('feedbackMessage');
    if (!feedbackEl) return;
    
    feedbackEl.style.display = 'block';
    
    if (correct) {
      feedbackEl.className = 'feedback-message feedback-success';
      feedbackEl.innerHTML = `
        <strong>Correct!</strong> ${this.getPositiveFeedback()}
      `;
      utils.showNotification('Correct answer! Great job! 🎉', 'success');
    } else {
      feedbackEl.className = 'feedback-message feedback-error';
      feedbackEl.innerHTML = `
        <strong>Incorrect.</strong> The correct answer is: ${state.currentQuestion.options[state.currentQuestion.correct]}
        <br><em>Hint: ${state.currentQuestion.hint}</em>
      `;
      utils.showNotification('Incorrect. Try reviewing the hint!', 'error');
    }
  },

  // Get positive feedback message
  getPositiveFeedback() {
    const messages = [
      'Excellent work!',
      'You\'re on fire!',
      'Keep it up!',
      'Outstanding!',
      'Brilliant!',
      'You\'re doing great!',
      'Perfect!'
    ];
    return utils.randomElement(messages);
  },

  // Load next question
  loadNext() {
    const question = this.getNextQuestion();
    this.displayQuestion(question);
  }
};
