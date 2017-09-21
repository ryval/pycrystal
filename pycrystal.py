import matplotlib.pyplot as plt
import numpy as np

from config import AFF

class SingleCrystal(object):

    def __init__(self, element, lat_const, lat_vect, bas_vect): 
        self.element = element
        self.lat_const = lat_const
        self.lat_vect = lat_vect
        self.bas_vect = bas_vect # Assuming basis vector in direct coordinates

    @property
    def element(self):
        return self._element

    @property
    def lat_const(self):
        return self._lat_const

    @property
    def lat_vect(self):
        return self._lat_vect

    @property
    def bas_vect(self):
        return self._bas_vect

    @element.setter
    def element(self, e):
        aff_data = [record for record in AFF if record[u'element'] == e]
        if not aff_data:
            raise Exception(u'element {} not supported'.format(e))
        else:
            record = aff_data[0]
            self._element = e
            self.a1 = float(record[u'a1'])
            self.a2 = float(record[u'a2'])
            self.a3 = float(record[u'a3'])
            self.a4 = float(record[u'a4'])
            self.b1 = float(record[u'b1'])
            self.b2 = float(record[u'b2'])
            self.b3 = float(record[u'b3'])
            self.b4 = float(record[u'b4'])
            self.c = float(record[u'c'])

    @lat_const.setter
    def lat_const(self, lc):
        if not isinstance(lc, float):
            raise Exception(u'lattice constant must be a float')
        else:
            self._lat_const = lc

    @lat_vect.setter
    def lat_vect(self, lv):
        self._lat_vect = np.array(lv)

    @bas_vect.setter
    def bas_vect(self, bv):
        self._bas_vect = np.array(bv)

    def reciprocal_vectors(self):
        a1 = self.lat_const * self.lat_vect[0]
        a2 = self.lat_const * self.lat_vect[1]
        a3 = self.lat_const * self.lat_vect[2]
        vol = np.dot(a1, np.cross(a2, a3))

        b1 = 2 * np.pi * (np.cross(a2, a3) / vol)
        b2 = 2 * np.pi * (np.cross(a3, a1) / vol)
        b3 = 2 * np.pi * (np.cross(a1, a2) / vol)
        return np.array([b1, b2, b3], dtype='f8')

    def atomic_form_factor(self, q):
        constants = [(self.a1, self.b1),
                     (self.a2, self.b2),
                     (self.a3, self.b3),
                     (self.a4, self.b4)]
        return np.sum([a * np.exp(-b * (q / 4.0 * np.pi)**2.0)
                       for a, b in constants], axis=0) + self.c

    def structure_factor(self, q_vect):
        return np.sum([self.atomic_form_factor(np.linalg.norm(q_vect, axis=-1)) * \
                       np.exp(-1j * np.dot(q_vect, bv)) for bv in self.bas_vect], axis=0)

    def get_xrd_pattern(self, hkl_min, hkl_max, decimals=1):
        q_vect = self.get_reciprocal_points(hkl_min, hkl_max, return_vectors=True)
        sf_sqd = np.absolute(self.structure_factor(q_vect))
        q_norm, idx, wts = self.get_reciprocal_points(hkl_min, hkl_max, decimals=decimals)
        return (q_norm, sf_sqd[idx] * wts)

    def plot_xrd_pattern(self, hkl_min, hkl_max):
        X, Y = self.get_xrd_pattern(hkl_min, hkl_max)
        plt.scatter(X, Y)
        plt.ylim(min(Y) - 0.1 * (max(Y) - min(Y)), max(Y) + 0.1 * (max(Y) - min(Y)))
        y_min, y_max = plt.ylim()
        for i, x in enumerate(X):
            plt.axvline(x=x, ymax=(Y[i] - y_min) / (y_max - y_min), linestyle='--')
        plt.show()

    def get_miller_indices(self, hkl_min, hkl_max):
        return np.mgrid[hkl_min:hkl_max+1:1,
                        hkl_min:hkl_max+1:1,
                        hkl_min:hkl_max+1:1].reshape(3, -1).T

    def get_reciprocal_points(self, hkl_min, hkl_max, return_vectors=False, decimals=1):
        hkls = self.get_miller_indices(hkl_min, hkl_max)
        if return_vectors:
            q_vects = np.array([np.dot(self.reciprocal_vectors(), hkl) for hkl in hkls])
            return q_vects[np.argsort(np.linalg.norm(q_vects, axis=1))][1:]
        else:
            return np.unique(np.around(sorted([np.linalg.norm(np.dot(self.reciprocal_vectors(), hkl))
                                               for hkl in hkls])[1:], decimals=decimals),
                             return_index=True,
                             return_counts=True)

class MixedCrystal(SingleCrystal): # Needs Work - what if ele1 and ele2 have diff lat_vects?
    def __init__(self, x, 
                 element1, element2, 
                 lat_const1, lat_const2,
                 lat_vect, # currently supporting only common lattice type
                 bas_vect1, bas_vect2,
                 ratio=1.0):
        
        self.crystal1 = SingleCrystal(element1, lat_const1, lat_vect, bas_vect1)
        self.crystal2 = SingleCrystal(element2, lat_const2, lat_vect, bas_vect2)
        self.lat_const = (1 - x) * lat_const1 + x * lat_const2 # Vergard's Law
        self.lat_vect = lat_vect
        self.ratio = ratio  # ratio is amount of ele1 compared to ele2 
        self.x = x # mixing parameter, 0.5 => fully mixed
        
    def structure_factor(self, q_vect):
        crystal1_msf = [np.sqrt(self.ratio) * (1 - self.x) * \
            self.crystal1.atomic_form_factor(np.linalg.norm(q_vect, axis=-1)) * \
            np.exp(-1j * np.dot(q_vect, bv)) for bv in self.crystal1.bas_vect]
        crystal2_msf = [self.x * self.crystal2.atomic_form_factor(np.linalg.norm(q_vect, axis=-1)) * \
            np.exp(-1j * np.dot(q_vect, bv)) for bv in self.crystal2.bas_vect]
        return np.sum(crystal1_msf + crystal2_msf, axis=0)