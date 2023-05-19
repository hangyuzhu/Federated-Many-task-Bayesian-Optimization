import os
import numpy as np
from SOP.ObjectiveFunctions import Ackley, Griewank, Weierstrass, Rastrigin, Rosenbrock, Schwefel, Sphere
from SOP.ObjectiveFunctions import MultiTaskTransform
from scipy.io import loadmat


def create_tasks_diff_func(dim, normalized=False):
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    # CI-H
    cih = loadmat(os.path.join(cur_dir_path, '50d/CI_H.mat'))
    transform1 = MultiTaskTransform(cih['Rotation_Task1'][0:dim, 0:dim], np.squeeze(cih['GO_Task1'])[0:dim])
    task1 = Griewank(x_lb=-100, x_ub=100, d=dim, normalized=normalized, transform=transform1)
    transform2 = MultiTaskTransform(cih['Rotation_Task2'][0:dim, 0:dim], np.squeeze(cih['GO_Task2'])[0:dim])
    task2 = Rastrigin(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform2)

    # CI-M
    cim = loadmat(os.path.join(cur_dir_path, '50d/CI_M.mat'))
    transform3 = MultiTaskTransform(cim['Rotation_Task1'][0:dim, 0:dim], np.squeeze(cim['GO_Task1'])[0:dim])
    task3 = Ackley(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform3)
    transform4 = MultiTaskTransform(cim['Rotation_Task2'][0:dim, 0:dim], np.squeeze(cim['GO_Task2'])[0:dim])
    task4 = Rastrigin(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform4)

    # CI-L
    cil = loadmat(os.path.join(cur_dir_path, '50d/CI_L.mat'))
    transform5 = MultiTaskTransform(cil['Rotation_Task1'][0:dim, 0:dim], np.squeeze(cil['GO_Task1'])[0:dim])
    task5 = Ackley(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform5)
    transform6 = None
    task6 = Schwefel(x_lb=-500, x_ub=500, d=dim, normalized=normalized, transform=transform6)

    # PI-H
    pih = loadmat(os.path.join(cur_dir_path, '50d/PI_H.mat'))
    transform7 = MultiTaskTransform(pih['Rotation_Task1'][0:dim, 0:dim], np.squeeze(pih['GO_Task1'])[0:dim])
    task7 = Rastrigin(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform7)
    transform8 = MultiTaskTransform(shift_mat=np.squeeze(pih['GO_Task2'])[0:dim])
    task8 = Sphere(x_lb=-100, x_ub=100, d=dim, normalized=normalized, transform=transform8)

    # PI-M
    pim = loadmat(os.path.join(cur_dir_path, '50d/PI_M.mat'))
    transform9 = MultiTaskTransform(pim['Rotation_Task1'][0:dim, 0:dim], np.squeeze(pim['GO_Task1'])[0:dim])
    task9 = Ackley(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform9)
    transform10 = None
    task10 = Rosenbrock(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform10)

    # PI-L
    pil = loadmat(os.path.join(cur_dir_path, '50d/PI_L.mat'))
    transform11 = MultiTaskTransform(pil['Rotation_Task1'][0:dim, 0:dim], np.squeeze(pil['GO_Task1'])[0:dim])
    task11 = Ackley(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform11)
    # TODO task 12 has different dimensions can be filtered out for convenience
    transform12 = MultiTaskTransform(pil['Rotation_Task2'][0:dim, 0:dim], np.squeeze(pil['GO_Task2'])[0:dim])
    task12 = Weierstrass(x_lb=-0.5, x_ub=0.5, d=dim, normalized=normalized, transform=transform12)

    # NI-H
    nih = loadmat(os.path.join(cur_dir_path, '50d/NI_H.mat'))
    # TODO task10 and task13 the same task
    transform13 = None
    task13 = Rosenbrock(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform13)
    transform14 = MultiTaskTransform(nih['Rotation_Task2'][0:dim, 0:dim], np.squeeze(nih['GO_Task2'])[0:dim])
    task14 = Rastrigin(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform14)

    # NI-M
    nim = loadmat(os.path.join(cur_dir_path, '50d/NI_M.mat'))
    transform15 = MultiTaskTransform(nim['Rotation_Task1'][0:dim, 0:dim], np.squeeze(nim['GO_Task1'])[0:dim])
    task15 = Griewank(x_lb=-100, x_ub=100, d=dim, normalized=normalized, transform=transform15)
    transform16 = MultiTaskTransform(nim['Rotation_Task2'][0:dim, 0:dim], np.squeeze(nim['GO_Task2'])[0:dim])
    task16 = Weierstrass(x_lb=-0.5, x_ub=0.5, d=dim, normalized=normalized, transform=transform16)

    # NI-L
    nil = loadmat(os.path.join(cur_dir_path, '50d/NI_L.mat'))
    transform17 = MultiTaskTransform(nil['Rotation_Task1'][0:dim, 0:dim], np.squeeze(nim['GO_Task1'])[0:dim])
    task17 = Rastrigin(x_lb=-50, x_ub=50, d=dim, normalized=normalized, transform=transform17)
    # TODO task6 and task18 are the same
    transform18 = None
    task18 = Schwefel(x_lb=-500, x_ub=500, d=dim, normalized=normalized, transform=transform18)

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8,
             task9, task10, task11, task12, task13, task14, task15, task16, task17, task18]
    return tasks
