import numpy as np


NGRID = 500
NEUTRALS = set([22, 2112])
MAXPARTICLES = 2000


def feta(pt, pz):
    return -np.log(np.tan(0.5 * np.arctan(np.abs(pt / pz)))) * np.sign(pz)


def ftheta(eta):
    return 2 * np.exp(np.exp(-eta))


def fphi(px, py):
    phi = np.arctan(py / px)
    if px < 0:
        if py > 0:
            return np.pi + phi
        else:
            return phi - np.pi
    else:
        return phi


def fpxyz(pt, eta, phi):
    px = np.cos(phi) * pt
    py = np.sin(phi) * pt
    pz = np.tan(ftheta(eta)) * pt
    return px, py, pz


class Record(object):
    def __init__(self, fpath):
        self.fpath = fpath
        self.f = open(fpath)

    def get_event(self):
        lines = []
        for l_ in self.f:
            l = l_.strip()
            if l.startswith('#'):
                continue
            if 'end' in l:
                break
            lines.append(l)
        if not lines:
            raise Exception('Input file (%s) is exhausted!'%self.fpath)
        return lines


class Particle(object):
    __slots__ = ['y', 'eta', 'phi', 'pt', 'm', 'pdgid', 'q']
    def __init__(self, line=None, vidx=None, eta=None, phi=None, pt=None, pdgid=None, q=None):
        self.y = vidx
        if line is not None:
            px, py, pz, self.m, self.pdgid = [float(l) for l in line.split()]
            self.pt = np.sqrt(px*px + py*py)
            self.eta = feta(self.pt, pz)
            self.phi = fphi(px, py)
        else:
            self.pt = pt
            self.eta = eta
            self.phi = phi
            self.pdgid = pdgid
            self.q = q

    @property
    def x(self):
        vlabel = self.y if self.q else -1 
        return np.array([self.pt, self.eta, self.phi, self.pdgid, vlabel]) 


class Grid(object):
    def __init__(self, nidx):
        self.idxs = list(range(nidx))
        self._ps = [[] for _ in self.idxs]
        self._p = set([])

    def clear(self):
        self._ps = [[] for _ in self.idxs]
        self._p = set([])

    def add(self, p):
        if abs(p.eta) > 5:
            return
        self._ps[p.y].append(p)

    def get_particles(self):
        pss = [[] for _ in self.idxs]
        hs = []
        for ps in self._ps:
            # loop through each interaction
            eta = [p.eta for p in ps]
            phi = [p.phi for p in ps]
            energy = [p.pt for p in ps]
            H, _, _ = np.histogram2d(eta, phi,
                                     bins=[np.linspace(-5, 5, NGRID), np.linspace(-np.pi, np.pi, NGRID)],
                                     weights=energy)
            hs.append(H)
        total_energy = sum(hs)
        concat = np.stack(hs, axis=0)
        non_zero = np.argwhere(total_energy > 0)
        for idx in non_zero:
            ieta, iphi = idx 
            sliced = concat[:, ieta, iphi]
            vidx = np.argmax(sliced)
            eta = (10 * ieta / NGRID) - 5
            phi = (2 * np.pi * iphi / NGRID) - np.pi
            pss[vidx].append(Particle(vidx=vidx, eta=eta, phi=phi, pt=np.sum(sliced), pdgid=0, q=0))
        return pss 

    def run(self, ps):
        for p in ps:
            self.add(p)
        return self.get_particles()


class Interaction(object):
    __slots__ = ['vidx', 'charged','neutral']
    def __init__(self, rec=None, vidx=-1, npu=1):
        self.charged = []
        self.neutral = []
        self.vidx = vidx
        if rec is None:
            return 
        for _ in range(npu):
            for l in rec.get_event():
                p = Particle(line=l, vidx=vidx)
                if abs(p.eta) > 5:
                    continue
                if p.pdgid in NEUTRALS:
                    p.q = 0
                    self.neutral.append(p)
                else:
                    p.q = 1
                    self.charged.append(p)

    @property
    def particles(self):
        return self.charged + self.neutral

    @property
    def x(self):
        return np.array([p.x for p in self.particles]).reshape(-1, 5)

    @property
    def y(self):
        return np.array([p.y for p in self.particles])

    def get_neutrals(self):
        n = self.neutral
        self.neutral = []
        return n


class Event(object):
    __slots__ = ['x','y','N']
    def __init__(self, hard_rec, pu_rec, npu, grid=None):
        hard = Interaction(hard_rec, 0)
        pus = [Interaction(pu_rec, i+1, 1) for i in range(npu)]

        ints = [hard] + pus

        if grid is not None:
            grid.clear()
            neutrals = []
            for i in ints:
                neutrals += i.get_neutrals()
            ns = grid.run(neutrals)
            hard.neutral = ns[0]
            for i,p in enumerate(pus):
                p.neutral = ns[i+1]

        x_list = [i.x for i in ints]
        self.x = np.concatenate(x_list, axis=0)

        # pt-ordering
        idx = np.argsort(self.x[:,0], axis=0)
        idx = np.flip(idx, axis=0)
        
        y_list = [i.y for i in ints]
        self.y = np.concatenate(y_list, axis=0)

        N = self.x.shape[0]
        self.N = N
        idx_full = idx
        if N > MAXPARTICLES:
            idx = idx[:MAXPARTICLES]
            N = MAXPARTICLES

        self.x = self.x[idx]
        self.y = self.y[idx]

        if N < MAXPARTICLES:
            self.x = np.pad(self.x, pad_width=((0, MAXPARTICLES-N), (0, 0)), constant_values=(0,0))
            self.y = np.pad(self.y, pad_width=((0, MAXPARTICLES-N),), constant_values=(-1,))

        self.N = N
