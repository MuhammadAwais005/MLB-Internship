# Student Performance Dashboard

A Jupyter notebook that analyzes scores in Python, Mathematics, Statistics, and Machine Learning for 20 students.

## Data Cleaning

No data-cleaning transformations were performed. The notebook loads the CSV as provided and calculates two analysis columns: each student's `Average` across the four subjects and a `Performance` category based on that average. These are derived fields, not cleaning steps.

## Visualizations

- **Bar chart:** compares each student's average score across all four subjects.
- **Histogram:** shows the distribution of student average scores.
- **Scatter plot:** compares Python and Machine Learning scores to show how the two subject scores vary together.
- **Pie chart:** shows the proportion of students in the Poor, Average, Good, and Excellent categories.
- **Box plot:** compares the spread and potential outliers of scores across the four subjects.

## Three Key Insights

1. Machine Learning has the highest class average at **82.6**, while Python has the lowest at **78.9**.
2. By the notebook's score bands, **10 students are Excellent**, 6 are Good, 3 are Average, and 1 is Poor. The average of the four-subject student averages is **80.4**.
3. **Laiba Khan** is the top performer across the four subjects, scoring 97 in Python, 95 in Mathematics, 98 in Statistics, and 99 in Machine Learning.

## Files

- `main.ipynb` - analysis and visualizations.
- `student_performance.csv` - source dataset.

## Requirements and Run

Install Python 3 and the notebook dependencies:

```bash
python -m pip install numpy pandas matplotlib jupyter
```

Open `main.ipynb` in Jupyter and run the cells from top to bottom. The notebook currently uses an absolute Windows path to load the CSV; to run it from another location, change that `pd.read_csv(...)` path to:

```python
data = pd.read_csv("student_performance.csv")
```
