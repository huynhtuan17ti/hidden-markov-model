import numpy as np

def forward(V: np.array, A: np.array, B:np.array, initial_prob: np.array):
    '''
        Dynamic formula:
            dp[t, x] = B[x] x sum(dp[t-1, i] x A[i][x]) for i = [1, M]
    ------------
    Parameters
    ------------
        - V: observation sequence
        - A: transition probabilities matrix
        - B: emission probabilities matrix
        - initial_prob: initial distribution matrix
    ------------
    Returns
    ------------
        - dp: result matrix
    '''
    T = V.shape[0] # number of time steps
    M = A.shape[0] # number of hidden states
    dp = np.zeros((T, M)) # [T, M]
    # base dynamic array
    dp[0, :] = initial_prob * B[:, V[0]]
 
    for t in range(1, T):
        for x in range(M):
            dp[t, x] = dp[t-1].dot(A[:, x]) * B[x, V[t]]
    return dp

def backward(V: np.array, A: np.array, B: np.array):
    '''
        Dynamic formula:
            dp[t, x] = B[x] x sum(dp[t+1, i] x A[x][i]) for i = [1, M]
    ------------
    Parameters
    ------------
        - V: observation sequence
        - A: transition probabilities matrix
        - B: emission probabilities matrix
    ------------
    Returns
    ------------
        - dp: result matrix
    '''
    T = V.shape[0] # number of time steps
    M = A.shape[0] # number of hidden states
    dp = np.zeros((T, M))
 
    # setting dp(T) = 1
    dp[T - 1] = np.ones((M))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(T - 2, -1, -1):
        for x in range(M):
            dp[t, x] = (dp[t+1] * B[:, V[t+1]]).dot(A[x, :])
    return dp

def viterbi(V: np.array, A: np.array, B:np.array, initial_prob: np.array):
    '''
        Dynamic formula:
            dp[t, x] = B[x] x max(dp[t-1, i] x A[i][x]) for i = [1, M]
    ------------
    Parameters
    ------------
        - V: observation sequence
        - A: transition probabilities matrix
        - B: emission probabilities matrix
        - initial_prob: initial distribution matrix
    ------------
    Returns
    ------------
        - result: optimal path
    '''
    T = V.shape[0] # number of time steps
    M = A.shape[0] # number of hidden states
    dp = np.zeros((T, M)) # [T, M]
    trace = np.zeros((T, M))
    # base dynamic array
    dp[0, :] = np.log(initial_prob * B[:, V[0]])
    
    for t in range(1, T):
        for x in range(M):
            probability = dp[t-1] + np.log(A[:, x]) + np.log(B[x, V[t]])
            trace[t-1, x] = np.argmax(probability)
            dp[t, x] = np.max(probability) 

    optimal_path = np.zeros(T)
    # find the most probable last hidden state
    last_state = np.argmax(dp[T-1, :])

    optimal_path[0] = last_state
    # backtracking
    cur_index = 1
    for i in range(T-2, -1, -1):
        optimal_path[cur_index] = trace[i, int(last_state)]
        last_state = trace[i, int(last_state)]
        cur_index += 1

    # reverse path
    optimal_path = np.flip(optimal_path, 0)
    return optimal_path

def baum_welch(V: np.array, A: np.array, B: np.array, initial_prob, n_iter=100):
    M = A.shape[0]
    T = len(V)

    for _ in range(n_iter):
        dpL = forward(V, A, B, initial_prob)
        dpR = backward(V, A, B)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(dpL[t, :].T, A) * B[:, V[t + 1]].T, dpR[t + 1, :])
            for x in range(M):
                numerator = dpL[t, x] * A[x, :] * B[:, V[t + 1]].T * dpR[t + 1, :].T
                xi[x, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = B.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, V == l], axis=1)

        B = np.divide(B, denominator.reshape((-1, 1)))
 
    return (A, B)