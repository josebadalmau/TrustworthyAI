# Initialize and fit the DKNN scorer on MNIST ID training features
dknn_mnist = DKNN(k=50)
dknn_mnist.fit(mnist_fit_features)

# Compute DKNN scores on the grid (higher = more OOD)
dknn_grid = dknn_mnist.compute_scores(torch.tensor(grid_points, dtype=torch.float32)).reshape(xx.shape)

# Plot DKNN score landscape
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DKNN OOD score landscape in 2D feature space")

cs = axes[0].contourf(xx, yy, dknn_grid, levels=30, cmap="RdYlGn_r")
axes[0].scatter(mnist_id_features[:, 0], mnist_id_features[:, 1],
                s=8, c="white", edgecolor="tab:blue", alpha=0.6, label="ID (0-4)")
axes[0].scatter(mnist_ood_features[:, 0], mnist_ood_features[:, 1],
                s=8, c="white", edgecolor="tab:red", alpha=0.6, label="OOD (5-9)")
axes[0].set_xlabel(r"$Z_1$")
axes[0].set_ylabel(r"$Z_2$")
axes[0].set_title("Score contours + test points")
axes[0].legend(loc="upper right")

cs2 = axes[1].contourf(xx, yy, dknn_grid, levels=30, cmap="RdYlGn_r")
axes[1].set_xlabel(r"$Z_1$")
axes[1].set_ylabel(r"$Z_2$")
axes[1].set_title("Score contours only")
plt.colorbar(cs2, ax=axes[1], label="DKNN OOD score (higher = more OOD)")
plt.tight_layout()
plt.show()
