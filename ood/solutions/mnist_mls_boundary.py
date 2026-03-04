# Extract weights and biases from the final linear layer
W = mnist_model.fc2.weight.detach().cpu().numpy()  # (num_classes, 2)
b = mnist_model.fc2.bias.detach().cpu().numpy()    # (num_classes,)

# Compute logits for each grid point: logits = z @ W.T + b
logits_grid = torch.tensor(grid_points @ W.T + b, dtype=torch.float32)

# Apply the MLS score (higher = more OOD)
mls_grid = mls(logits_grid).reshape(xx.shape)

# Plot MLS score landscape
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("MLS OOD score landscape in 2D feature space")

cs = axes[0].contourf(xx, yy, mls_grid, levels=30, cmap="RdYlGn_r")
axes[0].scatter(mnist_id_features[:, 0], mnist_id_features[:, 1],
                s=8, c="white", edgecolor="tab:blue", alpha=0.6, label="ID (0-4)")
axes[0].scatter(mnist_ood_features[:, 0], mnist_ood_features[:, 1],
                s=8, c="white", edgecolor="tab:red", alpha=0.6, label="OOD (5-9)")
axes[0].set_xlabel(r"$Z_1$")
axes[0].set_ylabel(r"$Z_2$")
axes[0].set_title("Score contours + test points")
axes[0].legend(loc="upper right")

cs2 = axes[1].contourf(xx, yy, mls_grid, levels=30, cmap="RdYlGn_r")
axes[1].set_xlabel(r"$Z_1$")
axes[1].set_ylabel(r"$Z_2$")
axes[1].set_title("Score contours only")
plt.colorbar(cs2, ax=axes[1], label="MLS OOD score (higher = more OOD)")
plt.tight_layout()
plt.show()
