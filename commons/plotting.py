import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_train_test_error(model_names:list[str], train_errors:list[np.number], test_errors:list[np.number]) -> None:
    positions = np.arange(len(model_names))
    bar_width = 0.3

    figure, ax = plt.subplots()
    ax.barh(positions, train_errors, bar_width, label='Train error')
    ax.barh(positions+bar_width, test_errors, bar_width, label='Test error')
    ax.set_yticks(positions+bar_width, model_names)
    ax.set_ylim([2*bar_width - 1, len(model_names)])
    ax.legend()
    ax.set_title('Train and test errors')

    plt.show()

def plot_confusion_matrices(model_names:list[str], labels:list[str], true_train_labels:list[Any], train_predictions:list[Any], true_test_labels:list[Any], test_predictions:list[Any], normalize:bool = False) -> None:
    
    figure = plt.figure(figsize=(7, 10), constrained_layout=True)
    figure.suptitle('Predictors confusion matrices', fontsize=14, fontweight='semibold')
    axes = figure.subfigures(len(model_names), 1, hspace=0.1)

    for row, row_axes in enumerate(axes):

        train_confusion_matrix = confusion_matrix(true_train_labels, train_predictions[row], normalize='true' if normalize else None)
        test_confusion_matrix = confusion_matrix(true_test_labels, test_predictions[row], normalize='true' if normalize else None)

        row_axes.suptitle(model_names[row], fontweight='semibold')
        col_axes = row_axes.subplots(1, 2)

        matrix_display = ConfusionMatrixDisplay(train_confusion_matrix, display_labels=labels)
        matrix_display.plot(ax=col_axes[0], xticks_rotation=45, cmap='Blues')
        matrix_display.ax_.set_title('Train samples')
        #matrix_display.im_.colorbar.remove()

        matrix_display = ConfusionMatrixDisplay(test_confusion_matrix, display_labels=labels)
        matrix_display.plot(ax=col_axes[1], xticks_rotation=45, cmap='Oranges')
        matrix_display.ax_.set_title('Test samples')
        #matrix_display.im_.colorbar.remove()

    plt.show()

def format_hyperparameters(hyperparameters_configuration:np.ndarray) -> list[Any]:
    return [
        hyperparameters_configuration[0].__name__,
        hyperparameters_configuration[1].__name__,
        hyperparameters_configuration[2].__name__,
        [str(condition) for condition in hyperparameters_configuration[3]],
        [str(condition) for condition in hyperparameters_configuration[4]]
    ]
