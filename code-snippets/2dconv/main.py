def conv2d(input_mat, kernel_mat):
    # dimensions of input matrix and kernel
    input_dim = len(input_mat)
    kernel_dim = len(kernel_mat)

    # check for valid dimensions
    if input_dim < kernel_dim:
        print("Error: Kernel dimension is greater than input dimension.")
        return []

    # calculate the dimensions of the output matrix
    output_dim = input_dim - kernel_dim + 1

    # initialize output matrix
    output_mat = [[0]*output_dim for _ in range(output_dim)]

    # apply the convolution operation
    for i in range(output_dim):
        for j in range(output_dim):
            for ki in range(kernel_dim):
                for kj in range(kernel_dim):
                    output_mat[i][j] += input_mat[i+ki][j+kj] * kernel_mat[ki][kj]

    return output_mat

def maxpool2d(input_mat, pool_size):
    # dimensions of input matrix
    input_dim = len(input_mat)

    # check for valid dimensions
    if input_dim < pool_size:
        print("Error: Pool size is greater than input dimension.")
        return []

    # calculate the dimensions of the output matrix
    output_dim = input_dim // pool_size

    # initialize output matrix
    output_mat = [[0]*output_dim for _ in range(output_dim)]

    # apply the max pooling operation
    for i in range(output_dim):
        for j in range(output_dim):
            output_mat[i][j] = max(max(input_mat[i*pool_size + ki][j*pool_size : j*pool_size + pool_size])
                                   for ki in range(pool_size))

    return output_mat

# Test the functions with an example
input_mat = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

kernel_mat = [
    [1, 0, -1],
    [0, 1, 0],
    [-1, 0, 1]
]

conv_result = conv2d(input_mat, kernel_mat)
print("Result of Convolution:")
for row in conv_result:
    print(row)

pool_result = maxpool2d(conv_result, 2)
print("\nResult of Max Pooling:")
for row in pool_result:
    print(row)
