"""Save an Actual vs Predicted scatter plot."""
import matplotlib.pyplot as plt


def plot_predictions(y_test, predictions, output_path, show=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, color="royalblue", s=80, label="Test students")
    low = min(min(y_test), min(predictions)) - 2
    high = max(max(y_test), max(predictions)) + 2
    ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
    ax.set(xlabel="Actual Average Score", ylabel="Predicted Average Score",
           title="Student Scores: Actual vs Predicted", xlim=(low, high),
           ylim=(low, high))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    
    
