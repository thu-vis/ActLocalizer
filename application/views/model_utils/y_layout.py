import numpy as np
import sys, os
from IPython import embed
from matplotlib import pyplot as plt
import pickle
import datetime
import os
import bisect
try:
    from DAGLayout import DAGLayout
except:
    None

decimals = 4
maxint = 1 << 31
min_non_valid_gap = 1
cluster_inner_dist_factor = 0.8

# 1. y-layout of multi mode
def solve_y_layout_adj(D, A, labels, orders):
    # if not os.path.exists('/tmp/VideoVis/'):
    #     os.mkdir('/tmp/VideoVis/')
    # time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # filepath = '/tmp/VideoVis/%s.pkl'%(time)
    orders = np.array(orders).T
    # pickle.dump({ 'D': D, 'A': A, 'labels': labels, 'orders': orders }, open(filepath, 'wb'))
        
    n_grids = 400
    alpha = 0
    beta = 85
    between_cluster_margin = 0.2
    inner_cluster_alpha = 0.8

    y = DAGLayout.DAGLayout(D, A, labels, n_grids, alpha, beta, between_cluster_margin, inner_cluster_alpha)
    y2 = 1 - np.array(y)

    y = np.zeros(y2.shape).astype(float)
    for i in range(orders.shape[0]):
        for j in range(orders.shape[1]):
            y[orders[i, j], j] = y2[i, j]

    # filepath = '/tmp/VideoVis/%s_layout.pkl'%(time)
    # print('write cache to %s'%(filepath))
    # pickle.dump({ 'D': D, 'A': A, 'labels': labels, 'y': y, 'orders': orders }, open(filepath, 'wb'))

    return y

# 2. y-layout of center mode
def layout_ctr(dist, order, grid_n = 100, alpha = 0, beta = 200):
    dist = dist.transpose()
    n_col = dist.shape[0]
    n_row = dist.shape[1]

    grid_step = 1.0 / grid_n
    max_sum = max(.5, dist.max())
    dist /= max_sum

    y = np.zeros((n_col, n_row))
    for i in range(n_col):
        f = np.ones((n_row, grid_n)) * maxint
        pre = []
        for j in range(n_row):
            pre.append([0] * grid_n)

        odr = order[i]
        for j in range(n_row):
            for k in range(grid_n):
                t = k * grid_step
                if i > 0:
                    ft = (t - y[i - 1, odr[j]]) ** 2 * alpha
                    if ft == 0:
                        ft = - beta * grid_step ** 2
                else:
                    ft = 0
                delta = dist[i, odr[j]]
                d = 1 - k * grid_step
                st = min(grid_n, k + 4)
                ed = grid_n
                for l in range(st, ed):
                    if j > 0:
                        d2 = f[odr[j - 1], l] + ft + (abs(d - delta) ** 2)
                    else:
                        d2 = ft + (abs(d - delta) ** 2)
                    if d2 < f[odr[j], k]:
                        f[odr[j], k] = d2
                        pre[odr[j]][k] = l
        
        last, k = odr[n_row - 1], 0
        for j in range(0, grid_n):
            if f[last][j] < f[last][k]:
                k = j
        cy = []
        for j in range(n_row - 1, -1, -1):
            cy.append(k)
            k = pre[odr[j]][k]
        cy = cy[::-1]
        cy = np.array(cy) * grid_step
        for j in range(n_row):
            y[i, odr[j]] = cy[j]

    y *= max_sum
    return y.transpose()

def solve_y_layout_ctr(D, order):
    layout = layout_ctr(D, order, 100)

    ret = np.zeros((D.shape[0] + 1, D.shape[1]))
    layout = 1 - layout
    for i in range(D.shape[0]):
        ret[i + 1] = layout[i]

    # x = np.array(range(ret.shape[1])).tolist()
    # plt.figure(figsize=(6, 2))
    # for r in ret:
    #     plt.plot(x, r)
    # plt.show()
    return layout

if __name__ == "__main__":
    # dists = np.array([[0.0, 0, 0, 0.0, 0, 0.0, 0.0, 0], [0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0], [0.0, 0, 0, 0.0, 0, 1, 0.5, 0], [1.0, 1, 0, 0.5, 1, 1, 1.0, 1]],dtype=np.float64)
    # order = [(0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)]
    
    dists = np.array([[0. , 0. , 0.5, 1. , 1. , 0.5, 0. ],
                      [1. , 1. , 0.5, 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1. , 0.5, 0. ],
                      [1. , 1. , 1. , 1. , 1. , 1. , 1. ]])
    order = [(2, 0, 1, 3), (2, 0, 1, 3), (2, 1, 0, 3), (1, 2, 0, 3), (1, 2, 0, 3), (1, 2, 0, 3), (1, 2, 0, 3)]
    res = solve_y_layout_ctr(dists, order)


# old versions by mosek 
def solve_y_layout_adj_v1(D, alpha):
    """
    Solve layout by Mosek Quadratic Optimization for adj situation.
    Params:
        D - distance matrix[n-1, w] def k = n-1
        alpha: arg in formula
    Return:
        Y - y position of points
    """
    # 1. Init paras and args
    # Parameters:
    #   Y  -  (k, w) matrix, adjust into a one-dimensional column vector
    #   Q  -  (kw, kw) quadratic term matrix between Y variables
    #   c  -  (kw, ) line term vector of Y variable
    k, w = D.shape
    assert k > 0 and w > 0
    num = k * w
    Y = np.zeros((k, w), dtype=np.int32)
    for i in range(k):
        for j in range(w):
            Y[i, j] = w * i + j
    Q = np.zeros((num, num))
    c = np.zeros((num, ))

    # 2. calculate Q, c
    # Using the method of four part superposition to calculate Q and c
    # Q0 needs to be a symmetric matrix, calculate Q0 = Q + QT
    # 2.1 i = 0, j = 0
    Q[Y[0, 0], Y[0, 0]] += 1
    c[Y[0, 0]] += (-2 * D[0, 0])
    # 2.2 i = 0, j > 0
    for j in range(1, w):
        Q[Y[0, j], Y[0, j]] += (1 + alpha)
        Q[Y[0, j], Y[0, j-1]] += (-2 * alpha)
        Q[Y[0, j-1], Y[0, j-1]] += alpha
        c[Y[0, j]] += (-2 * D[0, j])
    # 2.3 i > 0, j = 0
    for i in range(1, k):
        Q[Y[i, 0], Y[i, 0]] += 1
        Q[Y[i-1, 0], Y[i-1, 0]] += 1
        Q[Y[i-1, 0], Y[i, 0]] += (-2)
        c[Y[i, 0]] += (-2 * D[i, 0])
        c[Y[i-1, 0]] += 2 * D[i, 0]
    # 2.4 i > 0, j > 0
    for i in range(1, k):
        for j in range(1, w):
            Q[Y[i, j], Y[i, j]] += (1 + alpha)
            Q[Y[i-1, j], Y[i-1, j]] += 1
            Q[Y[i, j-1], Y[i, j-1]] += alpha
            Q[Y[i, j], Y[i-1, j]] += (-2)
            Q[Y[i, j], Y[i, j-1]] += (-2 * alpha)
            c[Y[i, j]] += (-2 * D[i, j])
            c[Y[i-1, j]] += 2 * D[i, j]
    Q = Q + Q.T

    # 3. cast Q and c into mosek type
    qsubi, qsubj, qval = [], [], []
    for i in range(num):
        for j in range(i+1):
            if Q[i, j] != 0:
                qsubi.append(i)
                qsubj.append(j)
                qval.append(Q[i, j])
    c = c.tolist()

    # 4. init constraints of mosek type
    # asub/aval  -  (num, ) means every para's constrain index and value 
    # save constrain matrix as csc matrix
    asub, aval = [], []
    for i in range(num):
        asub.append([])
        aval.append([])
    aindex = 0
    for i in range(1, k):
        for j in range(0, w):
            asub[Y[i,j]].append(aindex)
            asub[Y[i-1,j]].append(aindex)
            aval[Y[i,j]].append(1)
            aval[Y[i-1,j]].append(-1)
            aindex += 1
    
    # 4. Open MOSEK and create an environment and task
    # Refer to official documents: https://docs.mosek.com/latest/pythonapi/tutorial-qo-shared.html
    inf = 100
    # def streamprinter(text):
    #     sys.stdout.write(text)
    #     sys.stdout.flush()
    with mosek.Env() as env:
        # env.set_Stream(mosek.streamtype.log, streamprinter)
        with env.Task() as task:
            # task.set_Stream(mosek.streamtype.log, streamprinter)

            numvar = num
            numcon = (k - 1) * w

            # add empty constrain and vars
            task.appendcons(numcon)
            task.appendvars(numvar)

            # set variable bound
            for i in range(numvar):
                task.putvarbound(i, mosek.boundkey.lo, 0, inf)
            
            # set q: quadratic term
            task.putqobj(qsubi, qsubj, qval)

            # set c: line term
            for i in range(numvar):
                task.putcj(i, c[i])

            # set constrain
            for i in range(numvar):
                task.putacol(i, asub[i], aval[i])
            for i in range(numcon):
                task.putconbound(i, mosek.boundkey.lo, 0, inf)   

            # set task target
            task.putobjsense(mosek.objsense.minimize)

            # Optimize
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)

            # Output a solution
            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr, xx)
            result = np.array(xx).reshape((k, w))
            
    return result.round(decimals)

def solve_y_layout_adj_v2(D, alpha):
    """
    Solve layout by Mosek Quadratic Optimization for adj situation.
    Params:
        D - distance matrix[n-1, w] def k = n-1
        alpha: arg in formula
    Return:
        Y - y position of points
    """
    # 1. Init paras and args
    # Parameters:
    #   Y  -  (k, w) matrix, adjust into a one-dimensional column vector
    #   Q  -  (kw, kw) quadratic term matrix between Y variables
    #   c  -  (kw, ) line term vector of Y variable
    k, w = D.shape
    k = k + 1
    assert k > 0 and w > 0
    num = k * w
    Y = np.zeros((k, w), dtype=np.int32)
    for i in range(k):
        for j in range(w):
            Y[i, j] = w * i + j
    Q = np.zeros((num, num))
    c = np.zeros((num, ))

    # 2. calculate Q, c
    # Using the method of four part superposition to calculate Q and c
    # Q0 needs to be a symmetric matrix, calculate Q0 = Q + QT
    # 2.1 i = 0, j = 0
    # Q[Y[0, 0], Y[0, 0]] += 1
    # c[Y[0, 0]] += (-2 * D[0, 0])
    # 2.2 i = 0, j > 0
    for j in range(1, w):
        Q[Y[0, j], Y[0, j]] += alpha
        Q[Y[0, j], Y[0, j-1]] += (-2 * alpha)
        Q[Y[0, j-1], Y[0, j-1]] += alpha
    # 2.3 i > 0, j = 0
    for i in range(1, k):
        Q[Y[i, 0], Y[i, 0]] += 1
        Q[Y[i-1, 0], Y[i-1, 0]] += 1
        Q[Y[i-1, 0], Y[i, 0]] += (-2)
        c[Y[i, 0]] += (-2 * D[i-1, 0])
        c[Y[i-1, 0]] += 2 * D[i-1, 0]
    # 2.4 i > 0, j > 0
    for i in range(1, k):
        for j in range(1, w):
            Q[Y[i, j], Y[i, j]] += (1 + alpha)
            Q[Y[i-1, j], Y[i-1, j]] += 1
            Q[Y[i, j-1], Y[i, j-1]] += alpha
            Q[Y[i, j], Y[i-1, j]] += (-2)
            Q[Y[i, j], Y[i, j-1]] += (-2 * alpha)
            c[Y[i, j]] += (-2 * D[i-1, j])
            c[Y[i-1, j]] += 2 * D[i-1, j]
    Q = Q + Q.T

    # 3. cast Q and c into mosek type
    qsubi, qsubj, qval = [], [], []
    for i in range(num):
        for j in range(i+1):
            if Q[i, j] != 0:
                qsubi.append(i)
                qsubj.append(j)
                qval.append(Q[i, j])
    c = c.tolist()

    # 4. init constraints of mosek type
    # asub/aval  -  (num, ) means every para's constrain index and value 
    # save constrain matrix as csc matrix
    asub, aval = [], []
    for i in range(num):
        asub.append([])
        aval.append([])
    aindex = 0
    for i in range(1, k):
        for j in range(0, w):
            asub[Y[i,j]].append(aindex)
            asub[Y[i-1,j]].append(aindex)
            aval[Y[i,j]].append(1)
            aval[Y[i-1,j]].append(-1)
            aindex += 1
    
    # 4. Open MOSEK and create an environment and task
    # Refer to official documents: https://docs.mosek.com/latest/pythonapi/tutorial-qo-shared.html
    inf = 100
    # def streamprinter(text):
    #     sys.stdout.write(text)
    #     sys.stdout.flush()
    with mosek.Env() as env:
        # env.set_Stream(mosek.streamtype.log, streamprinter)
        with env.Task() as task:
            # task.set_Stream(mosek.streamtype.log, streamprinter)

            numvar = num
            numcon = (k - 1) * w

            # add empty constrain and vars
            task.appendcons(numcon)
            task.appendvars(numvar)

            # set variable bound
            for i in range(numvar):
                task.putvarbound(i, mosek.boundkey.lo, 0, inf)
            
            # set q: quadratic term
            task.putqobj(qsubi, qsubj, qval)

            # set c: line term
            for i in range(numvar):
                task.putcj(i, c[i])

            # set constrain
            for i in range(numvar):
                task.putacol(i, asub[i], aval[i])
            for i in range(numcon):
                task.putconbound(i, mosek.boundkey.lo, 0, inf)   

            # set task target
            task.putobjsense(mosek.objsense.minimize)

            # Optimize
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)

            # Output a solution
            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr, xx)
            result = np.array(xx).reshape((k, w))
    
    result = result - result.min()
    return result.round(decimals)

def solve_y_layout_ctr_v1(D, alpha=1):
    """
    Solve layout by Mosek Quadratic Optimization for center situation.
    Params:
        D - distance matrix[n-1, w] def k = n-1
        alpha: arg in formula
    Return:
        Y - y position of points
    """
    # 1. Init paras and args
    k, w = D.shape
    assert k > 0 and w > 0
    num = k * w
    Y = np.zeros((k, w), dtype=np.int32)
    for i in range(k):
        for j in range(w):
            Y[i, j] = w * i + j
    Q = np.zeros((num, num))
    c = np.zeros((num, ))

    # 2. calculate Q, c
    # 2.1 j = 0
    for i in range(k):
        Q[Y[i, 0], Y[i, 0]] += 1
        c[Y[i, 0]] += (- 2 * D[i, 0])
    # 2.2 j > 0
    for i in range(k):
        for j in range(1, w):
            Q[Y[i, j], Y[i, j]] += (1 + alpha)
            Q[Y[i, j-1], Y[i, j-1]] += alpha
            Q[Y[i, j], Y[i, j-1]] += (-2 * alpha)
            c[Y[i, j]] += (-2 * D[i, j])
    Q = Q + Q.T

    # 3. cast Q and c into mosek type
    qsubi, qsubj, qval = [], [], []
    for i in range(num):
        for j in range(i+1):
            if Q[i, j] != 0:
                qsubi.append(i)
                qsubj.append(j)
                qval.append(Q[i, j])
    c = c.tolist()

    # 4. Open MOSEK and create an environment and task
    # Refer to official documents: https://docs.mosek.com/latest/pythonapi/tutorial-qo-shared.html
    inf = 100
    # def streamprinter(text):
    #     sys.stdout.write(text)
    #     sys.stdout.flush()
    with mosek.Env() as env:
        # env.set_Stream(mosek.streamtype.log, streamprinter)
        with env.Task() as task:
            # task.set_Stream(mosek.streamtype.log, streamprinter)

            numvar = num
            # numcon = 0

            # add empty constrain and vars
            # task.appendcons(numcon)
            task.appendvars(numvar)

            # set variable bound
            for i in range(numvar):
                task.putvarbound(i, mosek.boundkey.lo, 0, inf)
            
            # set q: quadratic term
            task.putqobj(qsubi, qsubj, qval)

            # set c: line term
            for i in range(numvar):
                task.putcj(i, c[i])

            # # set constrain
            # for i in range(numvar):
            #     task.putacol(i, asub[i], aval[i])
            # for i in range(numcon):
            #     task.putconbound(i, mosek.boundkey.lo, 0, inf)   

            # set task target
            task.putobjsense(mosek.objsense.minimize)

            # Optimize
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)

            # Output a solution
            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr, xx)
            result = np.array(xx).reshape((k, w))
            
    return result.round(decimals)
