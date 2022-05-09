import hmm
import numpy as np

'''
    - The first dice is a balance dice, the probability of each side is 1 / 6 \
        With the second dice, the probability of 6 points is 0.5 and each of the rest is 0.1
    - At time t: \
        If Mr.Huy choose the balance dice, next time, the probability if this dice choosed again is 0.8 and 0.2 for the remainning
        If Mr.Huy choose the un-balance dice, next time, the probability if this dice choosed again is 0.3 and 0.7 for the balance

    - Everyone cannot know extractly which dice choosed by Mr.Huy \
        --> This is hidden states \
    - But they know the result after Mr.Huy roll the dice \
        --> This is observable states

    - Transition matrix \
        [[0.8, 0.2]
            [0.7, 0.3]]

    - Emission matrix \
        [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]

    - Initialize: \
        [0.5, 0.5] ????
'''

def solve():
    pass

if __name__ == '__main__':
    solve()