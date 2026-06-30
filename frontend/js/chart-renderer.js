// Chart Renderer Module - Chart.js wrapper for consistent chart rendering
import { utils } from './utils.js';

class ChartRenderer {
  constructor() {
    this.charts = {};
    this.defaultColors = {
      primary: '#4a6cf7',
      success: '#28a745',
      warning: '#ffc107',
      danger: '#dc3545',
      info: '#17a2b8',
      secondary: '#6c63ff'
    };
    
    this.emotionColors = {
      happy: '#28a745',
      bored: '#6c63ff',
      focused: '#17a2b8',
      confused: '#ffc107',
      neutral: '#6c757d',
      angry: '#dc3545',
      surprised: '#0ea5e9'
    };
    
    this.difficultyColors = {
      easy: '#28a745',
      medium: '#ffc107',
      hard: '#fd7e14'
    };

    this.subjectPalette = ['#2563eb', '#0891b2', '#16a34a', '#f59e0b', '#db2777', '#7c3aed', '#dc2626', '#0d9488'];
  }

  toTitleCase(value) {
    if (!value || typeof value !== 'string') return 'Unknown';
    return value
      .trim()
      .split(/\s+/)
      .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(' ');
  }

  // Destroy existing chart before creating new one
  destroyChart(chartId) {
    if (this.charts[chartId]) {
      this.charts[chartId].destroy();
      delete this.charts[chartId];
    }
  }

  // Subject Performance (Accuracy + Attempt Volume)
  renderSubjectPerformance(canvasId, subjects) {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      console.error(`Canvas ${canvasId} not found`);
      return null;
    }
    
    const ctx = canvas.getContext('2d');
    
    const rows = (Array.isArray(subjects) ? subjects : [])
      .filter(subject => subject && (subject.subject || subject.name))
      .map(subject => {
        const label = this.toTitleCase(subject.subject || subject.name);
        const total = Number(subject.total_questions || 0);
        const correct = Number(subject.correct_answers || 0);
        const accuracy = Number.isFinite(subject.accuracy)
          ? Number(subject.accuracy)
          : (total > 0 ? (correct / total) * 100 : 0);
        return {
          label,
          total,
          correct,
          accuracy: Number(accuracy.toFixed(1)),
          difficulty: subject.current_difficulty || 'N/A'
        };
      })
      .sort((a, b) => b.total - a.total || b.accuracy - a.accuracy);

    const labels = rows.map(r => r.label);
    const accuracyData = rows.map(r => r.accuracy);
    const volumeData = rows.map(r => r.total);
    const barColors = rows.map((_, i) => this.subjectPalette[i % this.subjectPalette.length] + 'CC');
    
    this.charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            type: 'bar',
            label: 'Accuracy %',
            data: accuracyData,
            yAxisID: 'y',
            backgroundColor: barColors,
            borderRadius: 8,
            borderSkipped: false,
            maxBarThickness: 38
          },
          {
            type: 'line',
            label: 'Target Accuracy (70%)',
            data: labels.map(() => 70),
            yAxisID: 'y',
            borderColor: '#2563eb',
            borderDash: [6, 6],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            fill: false
          },
          {
            type: 'line',
            label: 'Questions Attempted',
            data: volumeData,
            yAxisID: 'y1',
            borderColor: '#111827',
            backgroundColor: 'rgba(17, 24, 39, 0.15)',
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: '#ffffff',
            pointBorderWidth: 2,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              padding: 14,
              font: {
                size: 12,
                weight: '600',
                family: "'Inter', sans-serif"
              },
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          title: {
            display: false
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 12,
            titleFont: {
              size: 14,
              weight: '600'
            },
            bodyFont: {
              size: 13
            },
            callbacks: {
              label: function(context) {
                const row = rows[context.dataIndex];
                if (context.dataset.label === 'Accuracy %') {
                  return `Accuracy: ${context.parsed.y.toFixed(1)}% (${row.correct}/${row.total})`;
                }
                if (context.dataset.label === 'Target Accuracy (70%)') {
                  return 'Target: 70% benchmark';
                }
                return `Questions Attempted: ${context.parsed.y}`;
              },
              afterBody: function(context) {
                if (!context.length) return '';
                const row = rows[context[0].dataIndex];
                return `Difficulty: ${row.difficulty}`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: {
              color: 'rgba(0, 0, 0, 0.05)',
              drawBorder: false
            },
            ticks: {
              font: {
                size: 12,
                family: "'Inter', sans-serif"
              },
              callback: function(value) {
                return value + '%';
              },
              padding: 10
            },
            title: {
              display: true,
              text: 'Accuracy Rate',
              font: {
                size: 13,
                weight: '600'
              },
              padding: {top: 10, bottom: 0}
            }
          },
          y1: {
            beginAtZero: true,
            position: 'right',
            grid: {
              drawOnChartArea: false,
              drawBorder: false
            },
            ticks: {
              font: {
                size: 11,
                family: "'Inter', sans-serif"
              },
              precision: 0,
              padding: 8
            },
            title: {
              display: true,
              text: 'Questions',
              font: {
                size: 12,
                weight: '600'
              }
            }
          },
          x: {
            grid: {
              display: false,
              drawBorder: false
            },
            ticks: {
              font: {
                size: 12,
                family: "'Inter', sans-serif"
              },
              maxRotation: 40,
              minRotation: 0,
              padding: 10
            }
          }
        }
      }
    });
    
    return this.charts[canvasId];
  }

  // Emotion Distribution Pie/Donut Chart
  renderEmotionDistribution(canvasId, emotionData) {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      console.error(`Canvas ${canvasId} not found`);
      return null;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Handle both array and object formats
    let labels = [];
    let data = [];
    
    if (Array.isArray(emotionData)) {
      // Array format: [{ emotion: 'happy', count: 5 }, ...]
      labels = emotionData.map(item => item.emotion || 'Unknown');
      data = emotionData.map(item => item.count || 0);
    } else if (emotionData && typeof emotionData === 'object') {
      // Object format: { happy: 5, sad: 3, ... }
      labels = Object.keys(emotionData);
      data = Object.values(emotionData);
    } else {
      // No valid data
      console.warn('No valid emotion data provided');
      emotionData = {};
      labels = [];
      data = [];
    }
    
    const merged = labels.map((label, index) => ({ label, count: Number(data[index] || 0) }))
      .filter(item => item.count > 0)
      .sort((a, b) => b.count - a.count);

    labels = merged.map(item => item.label);
    data = merged.map(item => item.count);

    const colors = labels.map(emotion => this.emotionColors[String(emotion).toLowerCase()] || '#6c757d');
    
    this.charts[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels.map(l => {
          if (!l || typeof l !== 'string') return 'Unknown';
          return l.charAt(0).toUpperCase() + l.slice(1);
        }),
        datasets: [{
          data: data,
          backgroundColor: colors,
          borderWidth: 0,
          borderColor: 'transparent',
          spacing: 0,
          offset: 0,
          hoverOffset: 3,
          hoverBorderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '58%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 12,
              font: {
                size: 12,
                weight: '600',
                family: "'Inter', sans-serif"
              },
              usePointStyle: true,
              pointStyle: 'circle',
              generateLabels: function(chart) {
                const data = chart.data;
                if (data.labels.length && data.datasets.length) {
                  const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                  return data.labels.map((label, i) => {
                    const value = data.datasets[0].data[i];
                    const percentage = ((value / total) * 100).toFixed(1);
                    return {
                      text: `${label} (${percentage}%)`,
                      fillStyle: data.datasets[0].backgroundColor[i],
                      hidden: false,
                      index: i
                    };
                  });
                }
                return [];
              }
            }
          },
          title: {
            display: false
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 12,
            titleFont: {
              size: 14,
              weight: '600'
            },
            bodyFont: {
              size: 13
            },
            callbacks: {
              label: function(context) {
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = ((context.parsed / total) * 100).toFixed(1);
                return context.label + ': ' + context.parsed + ' times (' + percentage + '%)';
              }
            }
          }
        }
      }
    });
    
    return this.charts[canvasId];
  }

  // Difficulty Progression Bar Chart
  renderDifficultyProgression(canvasId, difficultyData) {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      console.error(`Canvas ${canvasId} not found`);
      return null;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Ensure difficultyData is an object
    if (!difficultyData || typeof difficultyData !== 'object') {
      difficultyData = {};
    }
    
    const difficulties = ['easy', 'medium', 'hard'];
    const data = difficulties.map(d => difficultyData[d] || 0);
    const colors = difficulties.map(d => this.difficultyColors[d] || '#94a3b8');
    
    const total = data.reduce((sum, val) => sum + val, 0);

    this.charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: difficulties.map(d => d.charAt(0).toUpperCase() + d.slice(1)),
        datasets: [{
          label: 'Questions Attempted',
          data: data,
          backgroundColor: colors.map(c => c + '90'),
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 8,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          title: {
            display: false
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 12,
            titleFont: {
              size: 14,
              weight: '600'
            },
            bodyFont: {
              size: 13
            },
            callbacks: {
              label: function(context) {
                const value = context.parsed.y;
                const percent = total > 0 ? ((value / total) * 100).toFixed(1) : '0.0';
                return `Questions: ${value} (${percent}%)`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: {
              color: 'rgba(0, 0, 0, 0.05)',
              drawBorder: false
            },
            ticks: {
              font: {
                size: 12,
                family: "'Inter', sans-serif"
              },
              stepSize: 1,
              padding: 10
            },
            title: {
              display: true,
              text: 'Number of Questions',
              font: {
                size: 13,
                weight: '600'
              },
              padding: {top: 10, bottom: 0}
            }
          },
          x: {
            grid: {
              display: false,
              drawBorder: false
            },
            ticks: {
              font: {
                size: 12,
                weight: '600',
                family: "'Inter', sans-serif"
              },
              padding: 10
            },
            title: {
              display: true,
              text: 'Difficulty Level',
              font: {
                size: 13,
                weight: '600'
              },
              padding: {top: 10, bottom: 0}
            }
          }
        }
      }
    });
    
    return this.charts[canvasId];
  }

  // Questions Timeline Area Chart
  renderQuestionsTimeline(canvasId, timelineData) {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      console.error(`❌ Canvas ${canvasId} not found`);
      return null;
    }
    
    console.log('🎨 Rendering timeline on canvas:', canvasId);
    
    const ctx = canvas.getContext('2d');
    
    // Ensure timelineData is valid
    if (!timelineData || typeof timelineData !== 'object') {
      console.warn('Timeline data invalid, using empty arrays');
      timelineData = { labels: [], correct: [], incorrect: [] };
    }
    
    console.log('📊 Chart data prepared:', {
      labels: timelineData.labels?.length || 0,
      correct: timelineData.correct?.length || 0,
      incorrect: timelineData.incorrect?.length || 0
    });
    
    const labels = Array.isArray(timelineData.labels) ? timelineData.labels : [];
    const correct = Array.isArray(timelineData.correct) ? timelineData.correct : [];
    const incorrect = Array.isArray(timelineData.incorrect) ? timelineData.incorrect : [];
    const pointCount = Math.max(labels.length, correct.length, incorrect.length);
    const safeLabels = Array.from({ length: pointCount }, (_, i) => labels[i] || `Day ${i + 1}`);
    const safeCorrect = Array.from({ length: pointCount }, (_, i) => Number(correct[i] || 0));
    const safeIncorrect = Array.from({ length: pointCount }, (_, i) => Number(incorrect[i] || 0));
    const accuracy = safeLabels.map((_, i) => {
      const c = Number(safeCorrect[i] || 0);
      const ic = Number(safeIncorrect[i] || 0);
      const total = c + ic;
      return total > 0 ? Number(((c / total) * 100).toFixed(1)) : 0;
    });

    this.charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: safeLabels,
        datasets: [{
          type: 'bar',
          label: 'Correct Answers',
          data: safeCorrect,
          borderColor: '#16a34a',
          backgroundColor: 'rgba(22, 163, 74, 0.82)',
          borderWidth: 1,
          borderRadius: 6,
          stack: 'answers',
          yAxisID: 'y'
        }, {
          type: 'bar',
          label: 'Incorrect Answers',
          data: safeIncorrect,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.82)',
          borderWidth: 1,
          borderRadius: 6,
          stack: 'answers',
          yAxisID: 'y'
        }, {
          type: 'line',
          label: 'Accuracy %',
          data: accuracy,
          yAxisID: 'y1',
          borderColor: '#1d4ed8',
          backgroundColor: 'rgba(29, 78, 216, 0.12)',
          tension: 0.3,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#fff',
          pointBorderWidth: 2,
          pointHoverBorderWidth: 3,
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              padding: 15,
              font: {
                size: 13,
                weight: '600',
                family: "'Inter', sans-serif"
              },
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          title: {
            display: false
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 12,
            titleFont: {
              size: 14,
              weight: '600'
            },
            bodyFont: {
              size: 13
            },
            callbacks: {
              label: function(context) {
                if (context.dataset.label === 'Accuracy %') {
                  return `Accuracy: ${context.parsed.y.toFixed(1)}%`;
                }
                return context.dataset.label + ': ' + context.parsed.y + ' questions';
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            stacked: true,
            grid: {
              color: 'rgba(0, 0, 0, 0.05)',
              drawBorder: false
            },
            ticks: {
              font: {
                size: 12,
                family: "'Inter', sans-serif"
              },
              stepSize: 1,
              padding: 10
            },
            title: {
              display: true,
              text: 'Questions Count',
              font: {
                size: 13,
                weight: '600'
              },
              padding: {top: 10, bottom: 0}
            }
          },
          y1: {
            beginAtZero: true,
            max: 100,
            position: 'right',
            grid: {
              drawOnChartArea: false,
              drawBorder: false
            },
            ticks: {
              callback: function(value) {
                return value + '%';
              },
              font: {
                size: 11,
                family: "'Inter', sans-serif"
              },
              padding: 8
            },
            title: {
              display: true,
              text: 'Accuracy %',
              font: {
                size: 12,
                weight: '600'
              }
            }
          },
          x: {
            stacked: true,
            grid: {
              display: false,
              drawBorder: false
            },
            ticks: {
              font: {
                size: 12,
                family: "'Inter', sans-serif"
              },
              maxRotation: 40,
              minRotation: 0,
              padding: 10
            },
            title: {
              display: true,
              text: 'Date',
              font: {
                size: 13,
                weight: '600'
              },
              padding: {top: 10, bottom: 0}
            }
          }
        }
      }
    });
    
    console.log('✅ Timeline chart created successfully');
    return this.charts[canvasId];
  }

  // Destroy all charts
  destroyAll() {
    Object.keys(this.charts).forEach(chartId => {
      this.destroyChart(chartId);
    });
  }
}

export const chartRenderer = new ChartRenderer();
