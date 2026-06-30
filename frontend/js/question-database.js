// Question database (will be moved to backend later)
export const questionDatabase = {
  elementary: [
    { id: 'elem_math_1', category: 'Mathematics', difficulty: 'easy', question: 'What is 5 + 3?', options: ['6', '7', '8', '9'], correct: 2, hint: 'Count on your fingers!' },
    { id: 'elem_math_2', category: 'Mathematics', difficulty: 'easy', question: 'What is 10 - 4?', options: ['5', '6', '7', '8'], correct: 1, hint: 'Subtract step by step' },
    { id: 'elem_sci_1', category: 'Science', difficulty: 'easy', question: 'What color is the sky on a clear day?', options: ['Green', 'Blue', 'Red', 'Yellow'], correct: 1, hint: 'Look up!' },
    { id: 'elem_sci_2', category: 'Science', difficulty: 'easy', question: 'How many legs does a spider have?', options: ['6', '8', '10', '12'], correct: 1, hint: 'Count carefully' },
    { id: 'elem_tech_1', category: 'Technology', difficulty: 'easy', question: 'What does a computer mouse do?', options: ['Types', 'Points', 'Prints', 'Calls'], correct: 1, hint: 'You move it around' },
    { id: 'elem_eng_1', category: 'Engineering', difficulty: 'easy', question: 'What holds up a bridge?', options: ['Water', 'Supports', 'Air', 'Clouds'], correct: 1, hint: 'Structural parts' }
  ],
  middle: [
    { id: 'mid_math_1', category: 'Mathematics', difficulty: 'medium', question: 'What is 15 × 4?', options: ['50', '55', '60', '65'], correct: 2, hint: 'Use multiplication' },
    { id: 'mid_sci_1', category: 'Science', difficulty: 'medium', question: 'What is photosynthesis?', options: ['Eating', 'Making food from sunlight', 'Breathing', 'Growing'], correct: 1, hint: 'Plants do this' },
    { id: 'mid_tech_1', category: 'Technology', difficulty: 'medium', question: 'What is HTML?', options: ['A programming language', 'Markup language', 'Database', 'Operating system'], correct: 1, hint: 'Used for web pages' },
    { id: 'mid_eng_1', category: 'Engineering', difficulty: 'medium', question: 'What is a blueprint?', options: ['A color', 'A plan/drawing', 'A tool', 'A material'], correct: 1, hint: 'Engineers use these' }
  ],
  high: [
    { id: 'high_math_1', category: 'Mathematics', difficulty: 'hard', question: 'What is the derivative of x²?', options: ['x', '2x', 'x²', '2x²'], correct: 1, hint: 'Power rule' },
    { id: 'high_sci_1', category: 'Science', difficulty: 'hard', question: 'What is the speed of light?', options: ['3×10⁸ m/s', '3×10⁶ m/s', '3×10⁴ m/s', '3×10² m/s'], correct: 0, hint: 'Very fast!' },
    { id: 'high_tech_1', category: 'Technology', difficulty: 'hard', question: 'What is an algorithm?', options: ['A number', 'Step-by-step procedure', 'A variable', 'A function'], correct: 1, hint: 'Instructions to solve a problem' },
    { id: 'high_eng_1', category: 'Engineering', difficulty: 'hard', question: 'What is tensile strength?', options: ['Compression', 'Resistance to pulling', 'Bending', 'Twisting'], correct: 1, hint: 'Material property' }
  ],
  college: [
    { id: 'col_math_1', category: 'Mathematics', difficulty: 'hard', question: 'What is the integral of 1/x?', options: ['x²', 'ln(x)', 'e^x', '1/x²'], correct: 1, hint: 'Natural logarithm' },
    { id: 'col_sci_1', category: 'Science', difficulty: 'hard', question: 'What is Heisenberg\'s Uncertainty Principle?', options: ['Position and momentum cannot both be known precisely', 'Energy is quantized', 'Time is relative', 'Entropy increases'], correct: 0, hint: 'Quantum mechanics' },
    { id: 'col_tech_1', category: 'Technology', difficulty: 'hard', question: 'What is Big O notation?', options: ['Time complexity', 'Space complexity', 'Algorithm efficiency', 'All of the above'], correct: 3, hint: 'Computer science concept' },
    { id: 'col_eng_1', category: 'Engineering', difficulty: 'hard', question: 'What is Finite Element Analysis?', options: ['Testing', 'Numerical method for solving equations', 'Design process', 'Manufacturing'], correct: 1, hint: 'Computational tool' }
  ]
};
