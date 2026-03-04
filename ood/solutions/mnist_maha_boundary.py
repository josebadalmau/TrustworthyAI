# Initialize and fit the Mahalanobis scorer on MNIST ID training features
maha_mnist = Mahalanobis()
maha_mnist.fit(mnist_fit_features, mnist_fit_labels.numpy())

# Compute Mahalanobis scores on the grid
# Note: the Mahalanobis class returns -min_distance, so lower values = more OOD
maha_grid = maha_mnist.compute_scores(torch.tensor(grid_points, dtype=torch.float32)).reshape(xx.shape)

# For visualization: negate so that higher = more OOD (consistent visual convention)
maha_grid_display = -maha_grid

# Plot Mahalanobis score landscape
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Mahalanobis OOD score landscape in 2D feature space")

cs = axes[0].contourf(xx, yy, maha_grid_display, levels=30, cmap="RdYlGn_r")
axes[0].scatter(mnist_id_features[:, 0], mnist_id_features[:, 1],
                s=8, c="white", edgecolor="tab:blue", alpha=0.6, label="ID (0-4)")
axes[0].scatter(mnist_ood_features[:, 0], mnist_ood_features[:, 1],
                s=8, c="white", edgecolor="tab:red", alpha=0.6, label="OOD (5-9)")
axes[0].set_xlabel(r"$Z_1$")
axes[0].set_ylabel(r"$Z_2$")
axes[0].set_title("Score contours + test points")
axes[0].legend(loc="upper right")

cs2 = axes[1].contourf(xx, yy, maha_grid_display, levels=30, cmap="RdYlGn_r")
axes[1].set_xlabel(r"$Z_1$")
axes[1].set_ylabel(r"$Z_2$")
axes[1].set_title("Score contours only")
plt.colorbar(cs2, ax=axes[1], label="Mahalanobis OOD score (higher = more OOD)")
plt.tight_layout()
plt.show()
