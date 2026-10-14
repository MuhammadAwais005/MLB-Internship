# Student Performance Analysis

A small Python data-analysis project that explores student scores across Python, Mathematics, Statistics, and Machine Learning. The analysis is implemented in a Jupyter notebook using pandas and NumPy.

## Project Contents

- `Student_Performance_Analysis.ipynb` - Notebook containing the data loading, analysis, and CSV export steps.
- `student_performance.csv` - Source dataset.
- `cleaned_student_data.csv` - CSV output written by the notebook. The current notebook exports the loaded DataFrame as-is; it does not perform additional cleaning transformations.

## Dataset

Each row represents one student. The dataset contains these columns:

| Column             | Description                              |
| ------------------ | ---------------------------------------- |
| `Student_ID`       | Student identifier                       |
| `Name`             | Student name                             |
| `Age`              | Student age                              |
| `Program`          | Student's academic program               |
| `Python`           | Python score                             |
| `Mathematics`      | Mathematics score                        |
| `Statistics`       | Statistics score                         |
| `Machine_Learning` | Machine Learning score                   |
| `Attendance`       | Attendance value recorded in the dataset |

## Analysis Performed

The notebook:

1. Loads the CSV data into a pandas DataFrame and previews the records.
2. Counts missing values in each column and checks the number of rows.
3. Calculates the mean score for each of the four subject columns. It also calculates the Python mean with a loop.
4. Displays five high-performing students using the subject score columns.
5. Lists students whose score is below the mean in each subject.
6. Counts the students and exports the DataFrame to `cleaned_student_data.csv`.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- NumPy

Install the Python packages with:

```bash
python -m pip install pandas numpy jupyter
```

## Run the Notebook

1. Open a terminal in the `Day_6` folder.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `Student_Performance_Analysis.ipynb` and run the cells from top to bottom.

### CSV Path Note

The notebook currently loads `student_performance.csv` using an absolute Windows path tied to the original author's machine. To run it from this project folder on another machine, change the `pd.read_csv(...)` line to:

```python
data = pd.read_csv("student_performance.csv")
```

The notebook writes `cleaned_student_data.csv` to its current working directory. Start Jupyter from the `Day_6` folder if you want the output saved there.
