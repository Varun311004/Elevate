// LocalStorage utility for persistence
export const storage = {
  // Save to localStorage
  save(key, data) {
    try {
      localStorage.setItem(key, JSON.stringify(data));
      return true;
    } catch (error) {
      console.error('Error saving to localStorage:', error);
      return false;
    }
  },

  // Load from localStorage
  load(key) {
    try {
      const data = localStorage.getItem(key);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Error loading from localStorage:', error);
      return null;
    }
  },

  // Remove from localStorage
  remove(key) {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.error('Error removing from localStorage:', error);
      return false;
    }
  },

  // Clear all localStorage
  clear() {
    try {
      localStorage.clear();
      return true;
    } catch (error) {
      console.error('Error clearing localStorage:', error);
      return false;
    }
  },

  // Save user session
  saveSession(user) {
    return this.save('elevate_user', user);
  },

  // Load user session
  loadSession() {
    return this.load('elevate_user');
  },

  // Save progress
  saveProgress(progress) {
    return this.save('elevate_progress', progress);
  },

  // Load progress
  loadProgress() {
    return this.load('elevate_progress');
  },

  // Save emotion history
  saveEmotionHistory(emotions) {
    return this.save('elevate_emotions', emotions);
  },

  // Load emotion history
  loadEmotionHistory() {
    return this.load('elevate_emotions');
  }
};
