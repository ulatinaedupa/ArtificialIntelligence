# Prerequisites - Foundational Skills for AI/ML

## Overview
Master the essential skills needed for artificial intelligence and machine learning. This comprehensive series covers Python programming, data science libraries, and development tools from the ground up.

## Project Structure
```
Prerequisitos/
├── Python Fundamentals (5-part series)
│   ├── Prerequisitos con Python - Parte 1.ipynb
│   ├── Prerequisitos con Python - Parte 2.ipynb
│   ├── Prerequisitos con Python - Parte 3.ipynb
│   ├── Prerequisitos con Python - Parte 4.ipynb
│   └── Prerequisitos con Python - Parte 5.ipynb
│
├── NumPy (2-part series)
│   ├── Prerequisitos Numpy - Parte 1.ipynb
│   └── Prerequisitos Numpy - Parte 2.ipynb
│
├── Pandas (2-part series)
│   ├── Prerequisitos con Pandas - Parte 1.ipynb
│   └── Prerequisitos con Pandas - Parte 2.ipynb
│
├── Matplotlib (2-part series)
│   ├── Prerequisitos con Matplotlib - Parte 1.ipynb
│   └── Prerequisitos con Matplotlib - Parte 2.ipynb
│
├── Development Tools
│   ├── Prerequisitos con Anacoda.ipynb
│   └── Prerequisitos Jupyter Notebook.ipynb
│
├── data/                    # Practice datasets
│   ├── Stock data: ADBE.csv, NVDA.csv, TSLA.csv
│   ├── credit_transactions.csv
│   ├── forestfires.csv
│   ├── fuel_consumption.csv
│   └── Pokemon.csv
│
├── imgs/                    # Tutorial images
└── scripts/                 # Python script examples
    ├── File operations
    ├── Error handling
    └── Module management
```

## Learning Path

### 🐍 Start Here: Python Fundamentals (5 Parts)

#### Part 1: Python Basics
- Variables and data types
- Basic operators
- String manipulation
- Input/output operations
- Comments and documentation

#### Part 2: Control Flow
- If/elif/else statements
- Loops (for, while)
- Break and continue
- List comprehensions
- Conditional expressions

#### Part 3: Data Structures
- Lists and tuples
- Dictionaries and sets
- Nested structures
- Common operations
- Choosing the right structure

#### Part 4: Functions & Modules
- Defining functions
- Parameters and return values
- Lambda functions
- Scope and namespaces
- Importing modules
- Creating custom modules

#### Part 5: Object-Oriented Programming
- Classes and objects
- Attributes and methods
- Inheritance
- Encapsulation
- Special methods (`__init__`, `__str__`)

### 🔢 NumPy: Numerical Computing (2 Parts)

#### Part 1: NumPy Fundamentals
- Array creation and manipulation
- Array indexing and slicing
- Array shapes and reshaping
- Broadcasting
- Basic operations

**How to Run:**
```bash
jupyter notebook "Prerequisitos Numpy - Parte 1.ipynb"
```

**Key Topics:**
```python
import numpy as np

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
random = np.random.rand(3, 3)

# Array operations
arr + 10
arr * 2
np.sqrt(arr)
```

#### Part 2: Advanced NumPy
- Mathematical functions
- Statistical operations
- Linear algebra
- Random number generation
- Array manipulation tricks

### 🐼 Pandas: Data Analysis (2 Parts)

#### Part 1: Pandas Basics
- Series and DataFrame
- Reading data (CSV, Excel)
- Data inspection
- Indexing and selection
- Basic operations

**How to Run:**
```bash
jupyter notebook "Prerequisitos con Pandas - Parte 1.ipynb"
```

**Key Topics:**
```python
import pandas as pd

# Load data
df = pd.read_csv('data/Pokemon.csv')

# Explore
df.head()
df.info()
df.describe()

# Select
df['Name']
df[['Name', 'Type 1']]
df[df['Attack'] > 100]
```

#### Part 2: Advanced Pandas
- Data cleaning
- Handling missing values
- GroupBy operations
- Merging and joining
- Pivot tables
- Time series basics

### 📊 Matplotlib: Visualization (2 Parts)

#### Part 1: Plotting Basics
- Line plots
- Scatter plots
- Bar charts
- Histograms
- Customizing plots (labels, titles, colors)

**How to Run:**
```bash
jupyter notebook "Prerequisitos con Matplotlib - Parte 1.ipynb"
```

**Key Topics:**
```python
import matplotlib.pyplot as plt

# Basic plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('My First Plot')
plt.show()

# Scatter plot
plt.scatter(x, y)
plt.show()
```

#### Part 2: Advanced Visualization
- Subplots and layouts
- Multiple plots
- Statistical plots
- Styling and themes
- Saving figures
- Interactive features

### 🛠️ Development Tools

#### Anaconda Setup
```bash
jupyter notebook "Prerequisitos con Anacoda.ipynb"
```

**Topics:**
- Installing Anaconda
- Creating environments
- Managing packages
- Conda vs pip
- Environment best practices

#### Jupyter Notebook
```bash
jupyter notebook "Prerequisitos Jupyter Notebook.ipynb"
```

**Topics:**
- Notebook interface
- Cell types (code, markdown)
- Keyboard shortcuts
- Magic commands
- Extensions and widgets

## Getting Started

### Installation

**Option 1: Anaconda (Recommended)**
```bash
# Download from https://www.anaconda.com/download
# Then create an environment
conda create -n ai-prereq python=3.11
conda activate ai-prereq
conda install jupyter numpy pandas matplotlib
```

**Option 2: pip**
```bash
python -m venv ai-prereq
source ai-prereq/bin/activate  # On Windows: ai-prereq\Scripts\activate
pip install jupyter numpy pandas matplotlib
```

### Start Learning
```bash
# Navigate to Prerequisitos folder
cd Prerequisitos

# Launch Jupyter
jupyter notebook

# Start with: Prerequisitos con Python - Parte 1.ipynb
```

## Practice Datasets

### Stock Data
- **ADBE.csv**: Adobe stock prices
- **NVDA.csv**: NVIDIA stock prices
- **TSLA.csv**: Tesla stock prices

**Use for:** Time series analysis, plotting, statistics

### Analysis Datasets
- **credit_transactions.csv**: Credit card transactions
- **forestfires.csv**: Forest fire occurrences
- **fuel_consumption.csv**: Vehicle fuel efficiency
- **Pokemon.csv**: Pokemon stats and attributes

**Use for:** Data manipulation, grouping, visualization

## Python Scripts (scripts/ folder)

### File Operations
```python
# Reading files
with open('file.txt', 'r') as f:
    content = f.read()
```

### Error Handling
- `asserting.py`: Using assertions
- `input_script_eh.py`: Input validation
- `value_error.py`: Exception handling

### Module Examples
- Creating reusable modules
- Importing custom code
- Package structure

## Learning Objectives

By completing these prerequisites, you will:
- ✅ Write clean, efficient Python code
- ✅ Manipulate data with NumPy arrays
- ✅ Analyze datasets using Pandas
- ✅ Create compelling visualizations
- ✅ Use Jupyter notebooks effectively
- ✅ Debug and handle errors
- ✅ Organize code in modules
- ✅ **Be ready for machine learning!**

## Recommended Study Order

**Week 1: Python Basics**
1. Python Part 1 (Basics)
2. Python Part 2 (Control Flow)
3. Python Part 3 (Data Structures)
4. Practice with scripts/

**Week 2: Advanced Python**
5. Python Part 4 (Functions)
6. Python Part 5 (OOP)
7. Setup tools (Anaconda, Jupyter)

**Week 3: NumPy**
8. NumPy Part 1
9. NumPy Part 2
10. Practice with numerical computations

**Week 4: Pandas**
11. Pandas Part 1
12. Pandas Part 2
13. Work with provided datasets

**Week 5: Visualization**
14. Matplotlib Part 1
15. Matplotlib Part 2
16. Create portfolio of visualizations

## Tips for Success

### Practice Regularly
- Code along with notebooks
- Modify examples
- Try different datasets
- Build small projects

### Use Resources
- Python documentation: https://docs.python.org/
- NumPy docs: https://numpy.org/doc/
- Pandas docs: https://pandas.pydata.org/docs/
- Matplotlib gallery: https://matplotlib.org/gallery/

### Debug Effectively
- Read error messages carefully
- Use print statements
- Test small pieces of code
- Ask for help when stuck

### Build Projects
- Analyze your own data
- Replicate interesting visualizations
- Automate repetitive tasks
- Share your work

## Common Challenges & Solutions

### Challenge: "I forget syntax"
**Solution:** Keep a cheat sheet, practice daily, use documentation

### Challenge: "Errors are confusing"
**Solution:** Read error messages from bottom up, Google the error, use Stack Overflow

### Challenge: "Don't know which library to use"
**Solution:**
- Data manipulation → Pandas
- Numerical computation → NumPy
- Visualization → Matplotlib
- File operations → Built-in Python

### Challenge: "Notebooks crash or freeze"
**Solution:** Restart kernel, save frequently, break code into smaller cells

## Next Steps

After completing prerequisites:
1. ✅ Start with **ProyectosEnClase/0 - Console Game Development Analysis**
2. ✅ Move to **Lab 0** for ML fundamentals
3. ✅ Progress through classification, regression, clustering
4. ✅ Advance to deep learning and AI applications

## Quick Reference

### Essential Imports
```python
# Always start notebooks with:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Common Operations
```python
# Load data
df = pd.read_csv('data/Pokemon.csv')

# Quick stats
df.describe()
df.info()

# Plot
df['Attack'].hist()
plt.show()

# Filter
strong = df[df['Attack'] > 100]

# Group
df.groupby('Type 1')['Attack'].mean()
```

### Jupyter Shortcuts
- `Shift + Enter`: Run cell
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell
- `M`: Markdown cell
- `Y`: Code cell
- `Tab`: Autocomplete

## Assessment Checklist

Before moving to ML projects, ensure you can:
- [ ] Write functions and classes in Python
- [ ] Create and manipulate NumPy arrays
- [ ] Load and clean data with Pandas
- [ ] Create various plot types with Matplotlib
- [ ] Handle file I/O operations
- [ ] Debug common errors
- [ ] Use Jupyter notebooks efficiently
- [ ] Understand basic statistics
- [ ] Work with CSV files
- [ ] Visualize data insights

## Get Help
- Review notebook markdown cells
- Check Python documentation
- Practice with provided datasets
- Experiment and break things (in a safe environment!)
- Move to projects when comfortable

**Remember:** Everyone starts here. Take your time, practice consistently, and you'll be building AI models before you know it!
