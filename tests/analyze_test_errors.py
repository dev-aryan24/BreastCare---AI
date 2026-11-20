
def create_overlay(
    image,
    ground_truth,
    prediction,
    save_path,
    title, np, plt
):
    height, width = ground_truth.shape

    ground_truth_rgba = np.zeros(
        (height, width, 4),
        dtype=np.float32
    )

    prediction_rgba = np.zeros(
        (height, width, 4),
        dtype=np.float32
    )

    comparison_rgba = np.zeros(
        (height, width, 4),
        dtype=np.float32
    )

    ground_truth_rgba[ground_truth > 0] = [
        1.0, 0.0, 0.0, 0.50
    ]

    prediction_rgba[prediction > 0] = [
        0.0, 1.0, 1.0, 0.55
    ]

    comparison_rgba[ground_truth > 0] = [
        1.0, 0.0, 0.0, 0.50
    ]

    comparison_rgba[prediction > 0] = [
        0.0, 1.0, 1.0, 0.55
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(image)
    axes[1].imshow(ground_truth_rgba)
    axes[1].set_title("Ground Truth - Red")

    axes[2].imshow(image)
    axes[2].imshow(prediction_rgba)
    axes[2].set_title("Prediction - Cyan")

    axes[3].imshow(image)
    axes[3].imshow(comparison_rgba)
    axes[3].set_title("Comparison - Red / Cyan")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(title)

    plt.tight_layout()

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)
