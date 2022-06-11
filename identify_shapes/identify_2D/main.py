import time

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly.base import partial_tensor_to_vec, tensor_to_vec

from hierarchical_tucker_regression import HierarchicalTuckerRegressor
from gen_2d_shape import gen_image

# Parameter of the experiment
image_height = 25
image_width = 25

# shape of the images
patterns = ['rectangle', 'swiss', 'circle', 'triangle', 'ring']
# patterns = ['swiss']

# ranks to test
ranks = [1, 2, 3, 4, 5]

# Generate random samples
X = np.random.normal(0, 1, (1000, image_height, image_width))

# Parameters of the plot, deduced from the data
n_rows = len(patterns)
n_columns = len(ranks) + 1

# Plot the three images
fig = plt.figure()

t0 = time.time()
for i, pattern in enumerate(patterns):

    print('fitting pattern n.{}'.format(i))

    # Generate the original image
    weight_img = gen_image(region=pattern, image_height=image_height, image_width=image_width)
    weight_img = tl.tensor(weight_img)

    # Generate the labels
    y = tl.dot(partial_tensor_to_vec(X, skip_begin=1), tensor_to_vec(weight_img))

    # Plot the original weights
    ax = fig.add_subplot(n_rows, n_columns, i * n_columns + 1)
    ax.imshow(tl.to_numpy(weight_img), cmap=plt.cm.OrRd, interpolation='nearest')
    ax.set_axis_off()
    if i == 0:
        ax.set_title('Original\nweights')

    for j, rank in enumerate(ranks):
        print('fitting for rank = {}'.format(rank))

        # Create a tensor Regressor estimator
        estimator = HierarchicalTuckerRegressor(
            dims=[image_width, image_height],
            ht_ranks=[1, rank, rank],
            tol=10e-7,
            n_iter_max=100,
            verbose=0
        )

        # Fit the estimator to the data
        estimator.fit(X, y)
        print('mse tensor for {}:'.format(pattern),
              format(((weight_img - estimator.weight_tensor_) ** 2).mean(axis=None), '0.7f'))

        ax = fig.add_subplot(n_rows, n_columns, i * n_columns + j + 2)
        ax.imshow(tl.to_numpy(estimator.weight_tensor_), cmap=plt.cm.OrRd, interpolation='nearest')
        ax.set_axis_off()

        if i == 0:
            ax.set_title('Core Tensor\nrank = [{}, {}]'.format(rank, rank))

t1 = time.time() - t0
print('executed time:', t1)

plt.suptitle("Hierarchical Tucker tensor regression")
plt.show()
