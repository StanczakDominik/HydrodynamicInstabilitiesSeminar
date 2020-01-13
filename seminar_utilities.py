from scipy.integrate import solve_ivp

import numpy as np
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from IPython.display import HTML

import pathlib
import ipywidgets
from pyro import io

# import matplotlib
# matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
import warnings
warnings.filterwarnings('ignore')

def cons_to_prim(U, gamma, ivars, myg):
    """ convert an input vector of conserved variables to primitive variables """

    q = myg.scratch_array(nvar=ivars.nq)

    q[:, :, ivars.irho] = U[:, :, ivars.idens]
    q[:, :, ivars.iu] = U[:, :, ivars.ixmom]/U[:, :, ivars.idens]
    q[:, :, ivars.iv] = U[:, :, ivars.iymom]/U[:, :, ivars.idens]

    e = (U[:, :, ivars.iener] -
         0.5*q[:, :, ivars.irho]*(q[:, :, ivars.iu]**2 +
                                  q[:, :, ivars.iv]**2))/q[:, :, ivars.irho]

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix+ivars.naux),
                          range(ivars.irhox, ivars.irhox+ivars.naux)):
            q[:, :, nq] = U[:, :, nu]/q[:, :, ivars.irho]

    return q



class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        try:
            self.idens = myd.names.index("density")
        except ValueError:
            pass
        try:
            self.ixmom = myd.names.index("x-momentum")
        except ValueError:
            pass
        try:
            self.iymom = myd.names.index("y-momentum")
        except ValueError:
            pass
        try:
            self.iener = myd.names.index("energy")
        except ValueError:
            pass

        # if there are any additional variable, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 4
        if self.naux > 0:
            self.irhox = 4
        else:
            self.irhox = -1

        # primitive variables
        self.nq = 4 + self.naux

        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.ip = 3

        if self.naux > 0:
            self.ix = 4   # advected scalar
        else:
            self.ix = -1



def pendulum(initial_theta = np.pi / 2, maxtime=10, max_step = np.inf):
    def f(t, y, g = 9.81, l=1):
        theta, omega = y
        return omega, -g * np.sin(theta)/ l

    sol = solve_ivp(f, (0, maxtime),
                    (initial_theta, 0),
                    dense_output=True,
                    max_step = max_step,
                    )

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(( -1.2, 1.2))
    ax.set_ylim((-1.5, 1.5))
    ax.set_aspect('equal')

    line, = ax.plot([], [], "o-", lw=2)



    def init():
        line.set_data([], [])
        return (line,)
    def animate(t):
        theta = sol.sol(t)[0]
        x = np.sin(theta)
        y = -np.cos(theta)
        line.set_data([0, x], [0, y])
        return (line,)

    plt.close()
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=np.arange(0, maxtime, 0.01), interval=10, 
                                   blit=True)
    return HTML(anim.to_html5_video())

def images(
    pathstring = "kh_*.h5",
    directory = "/home/dominik/FUW/Semestr3/PrezentacjaNiestabilności/pyro2/backup2",
    interval = 2000,
    figsize = (20, 20),
):
    p = pathlib.Path(directory)
    frames = sorted(list(p.glob(pathstring)))[::1]
    plotfile_name = str(frames[0])
    sims = [io.read(str(frame)) for frame in frames]
    fig = plt.figure(figsize=figsize, dpi=200, facecolor='w')
    
    slider = ipywidgets.IntSlider(min=1, max=len(frames)-1, step=1, continuous_update=True)
    play = ipywidgets.Play(min=1, max=len(frames)-1, interval=interval)

    display(play)
    ipywidgets.jslink((play, 'value'), (slider, 'value'))
    
    def interactive(i):
        sims[i].dovis()
        plt.show()
    return ipywidgets.interactive(interactive, i=slider)

def images2(
    pathstring = "kh_*.h5",
    directory = "/home/dominik/FUW/Semestr3/PrezentacjaNiestabilności/pyro2/backup2",
    interval = 2000,
    figsize = (15, 10),
    fieldname="rho",
    cmap = "viridis",
):
    p = pathlib.Path(directory)
    frames = sorted(list(p.glob(pathstring)))[::1]
    plotfile_name = str(frames[0])
    myg = io.read(plotfile_name).cc_data.grid

    def get_field(file, fieldname):
        sim = io.read(file)
        if fieldname == "vort":
            u = sim.cc_data.get_var("x-velocity")
            v = sim.cc_data.get_var("y-velocity")

            myg = sim.cc_data.grid

            vort = myg.scratch_array()
            divU = myg.scratch_array()

            vort.v()[:, :] = \
                 0.5*(v.ip(1) - v.ip(-1))/myg.dx - \
                 0.5*(u.jp(1) - u.jp(-1))/myg.dy
            return vort
        else:
            ivars = Variables(sim.cc_data)

            # access gamma from the cc_data object so we can use dovis
            # outside of a running simulation.
            gamma = sim.cc_data.get_aux("gamma")

            q = cons_to_prim(sim.cc_data.data, gamma, ivars, sim.cc_data.grid)
            field = eval(f"q[:, :, ivars.i{fieldname}]")
            return field


    fields = [get_field(str(frame), fieldname) for frame in frames]
    v = fields[0]

    fig, ax = plt.subplots(figsize=figsize, dpi=200, facecolor='w')
    img = ax.imshow(np.transpose(v.v()),
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                    cmap=cmap)
    ax.set_title(fieldname)
    time = plt.figtext(0.05, 0.0125, "snapshot = 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def init():
        img.set_array([[]])
        time.set_text("snapshot = 0")
        return img,
    def animate(i):
        img.set_array(np.transpose(fields[i]))
        time.set_text(f"snapshot = {i}")
        return img,

    plt.close()
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(fields), interval=interval, 
                                   blit=True)
    return HTML(anim.to_html5_video())

