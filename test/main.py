import numpy as np
from PIL import Image
from d_star.d_star import DStar

np.random.seed(0)

STEP = 30

paths = np.load('all_paths.npy')
paths[:, 0] = paths[:, 0] / 4
paths[:, 1] = paths[:, 1] / STEP
paths[:, 2] = paths[:, 2] / STEP

ped_time = paths[:,0]
ped_x = paths[:,1]
ped_y = paths[:,2]

obs = Image.open('obstacles.png')
obs = (~np.array(obs).sum(axis=2).astype(bool)).astype(int)

grid = np.zeros((int(1080 / STEP), int(1920 / STEP)), dtype=int)

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        x1, x2 = i * STEP, (i + 1) * STEP
        y1, y2 = j * STEP, (j + 1) * STEP
        if obs[x1 : x2, y1 : y2].sum() > STEP * STEP / 2:
            grid[i, j] = 1

x_start = 34 # from 0 to 35
y_start = 4 # from 0 to 63
x_goal = 25 # from 0 to 35
y_goal = 20 # from 0 to 63
start_time = 0 # from 0 to 30000


def update_cells(pf, x, y, time, block):
    t = ped_time == time
    ped_x_time = ped_x[t]
    ped_y_time = ped_y[t]
    ind_x = (ped_x_time >= x - 1) * (ped_x_time <= x + 1)
    ind_y = (ped_y_time >= y - 1) * (ped_y_time <= y + 1)
    ind = ind_x * ind_y

    for i, j in zip(ped_x_time[ind], ped_y_time[ind]):
        pf.update_cell(int(i), int(j), -1 if block else 0)


tmp = grid.copy()
pf = DStar(x_start, y_start, x_goal, y_goal)

for i, row in enumerate(grid):
    for j, cell in enumerate(row):
        if cell == 1:
            pf.update_cell(i, j, -1)

pf.replan()

time = start_time

x_cur, y_cur = x_start, y_start

while x_cur != x_goal or y_cur != y_goal:
    time += 1

    pf.update_start(x_cur, y_cur)
    tmp[x_cur, y_cur] = 7

    update_cells(pf, x_cur, y_cur, time, block=True)
    print(x_cur, y_cur)

    if not pf.replan():
        break

    x_prev, y_prev = x_cur, y_cur
    x_cur, y_cur = pf.get_path()[1].x, pf.get_path()[1].y

    update_cells(pf, x_prev, y_prev, time, block=False)

print(time - start_time)

print(tmp[x_goal:x_start+1, y_start:y_goal+1])