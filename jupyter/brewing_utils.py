import os
import sys
import argparse
import glob
import math
import importlib
import itertools
import time

from matplotlib import pyplot as plt
from matplotlib import cm

import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import scipy.spatial as spatial
import scipy.special as special
import numpy as np
import pandas as pd

import tifffile as tf

def clean_zeroes(img):
    dim = img.ndim
    orig_size = img.size

    cero = list(range(2*dim))

    for k in range(dim):
        ceros = np.all(img == 0, axis = (k, (k+1)%dim))

        for i in range(len(ceros)):
            if(~ceros[i]):
                break
        for j in range(len(ceros)-1, 0, -1):
            if(~ceros[j]):
                break
        cero[k] = i
        cero[k+dim] = j+1

    img = img[cero[1]:cero[4], cero[2]:cero[5], cero[0]:cero[3]]

    print(round(100-100*img.size/orig_size),'% reduction from input')

    return img


def normalize_density(img, adjust_by):
    resol = 2**(img.dtype.itemsize*8)
    npz = np.arange(resol, dtype=img.dtype)

    for i in range(len(npz)):
        aux = round(adjust_by[0]*npz[i]*npz[i] + adjust_by[1]*npz[i] + adjust_by[2])
        if aux < resol and aux > 0:
            npz[i] = int(aux)
        elif aux >= resol:
            npz[i] = resol - 1
        else:
            npz[i] = 0

    with np.nditer(img, flags=['external_loop'], op_flags=['readwrite']) as it:
        for x in it:
            x[...] = npz[x]

    return img


def misc_cleaning(img, sigma=3, thr1=55, ero=(7,7,7), dil=(5,5,5), thr2=30, op=(1,11,11)):
    blur = ndimage.gaussian_filter(img, sigma=sigma, mode='constant', truncate=3, cval=0)
    img[blur < thr1] = 0
    print('Gaussian blurred!')

    blur = ndimage.grey_erosion(img, mode='constant', size=ero)
    print('Eroded!')

    blur = ndimage.grey_dilation(blur, mode='constant', size=dil)
    print('Dilated!')

    img[blur < thr2] = 0
    blur = ndimage.grey_opening(img, mode='constant', size=op)
    print('Opened!')

    img[blur < thr2-10] = 0
    img = clean_zeroes(img)

    return img

def separate_pruned_spikes(dst, bname, img, cutoff = 1e-2, flex=2):

    labels,num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    print(num,'components')
    regions = ndimage.find_objects(labels)

    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    sz_hist = np.sum(hist)
    print('hist', hist)
    print('size =',sz_hist)

    argsort_hist = np.argsort(hist)[::-1]

    for j in range(len(regions)):
        i = argsort_hist[j]
        r = regions[i]
        if(hist[i]/sz_hist > cutoff):
            z0,y0,x0,z1,y1,x1 = r[0].start,r[1].start,r[2].start,r[0].stop,r[1].stop,r[2].stop
            mask = labels[r]==i+1
            box = img[r].copy()
            box[~mask] = 0
            mass = 1.0/np.sum(box)
            grow = np.arange(box.shape[0], dtype = 'float64')
            grow[0] = np.sum(box[0,:,:])

            for k in range(1,len(grow)):
                zmass = np.sum(box[k,:,:])
                grow[k] = grow[k-1] + zmass

            if grow[-1] != np.sum(box):
                print('grow[-1] != np.sum(box)', j, 'args.in_tiff')
                break

            grow = grow*mass
            logdiff = np.abs(np.ediff1d(np.gradient(np.log(grow))))
            critic = []

            for k in range(len(logdiff)-1,0,-1):
                if(logdiff[k] > 1e-6):
                    critic.append(k)
                    if(len(critic) > flex):
                        break

            if(np.sum(np.ediff1d(critic) == -1) == flex):
                k = critic[0]+1

            print('{} (x,y,z)=({},{},{}), (w,h,d)=({},{},{})'.format(j,x0,y0,z0,box.shape[2],box.shape[1],box.shape[0]))
            print(box.shape[0], critic, logdiff[-1])

            if( k+1 < box.shape[0]):
                print('Reduced from',box.shape[0],'to',k)
                box = box[:k,:,:]

            tf.imwrite('{}{}_l{}_x{}_y{}_z{}.tif'.format(dst,bname,j,x0,y0,z0),box,photometric='minisblack',compress=3)

    return 0

def seed_template(dst, img, seed, loc, size, padding=7, dil=4, thr=120, op=5, ero=3, sigma= 3, bname='seed', w=False):
    x,y,z = loc
    w,h,d = size

    pad_array = np.array([padding,padding,padding])

    for i in range(3):
        if loc[i] < padding:
            pad_array[i] = loc[i]

    pad_array[0], pad_array[-1] = pad_array[-1], pad_array[0]

    if z+d+pad_array[0] > img.shape[0]:
        pad_array[0] -= (z+d+pad_array[0]) -  img.shape[0]

    if y+h+pad_array[1] > img.shape[1]:
        pad_array[1] -= (y+h+pad_array[1]) -  img.shape[1]

    if x+w+pad_array[2] > img.shape[2]:
        pad_array[2] -= (x+w+pad_array[2]) -  img.shape[2]

    padded = np.zeros((np.array(seed.shape) + 2*pad_array)).astype('uint8')
    foo = np.array(seed.shape) + pad_array
    padded[ pad_array[0]:foo[0], pad_array[1]:foo[1], pad_array[2]:foo[2] ] = seed

    for i in range(dil):
        padded = ndimage.grey_dilation(padded,
                                       structure=(ndimage.generate_binary_structure(padded.ndim, 1)),
                                       mode='constant', cval=0)

    iso_seed = img[(z-pad_array[0]):(z+d+pad_array[0]),
                   (y-pad_array[1]):(y+h+pad_array[1]),
                   (x-pad_array[2]):(x+w+pad_array[2])].copy()

    padmask = padded > 50
    iso_seed = np.where(padmask, iso_seed, 0)
    iso_seed[iso_seed < thr] = 0
    iso_seed = ndimage.grey_opening(iso_seed, size=(dil,dil,dil), mode='constant', cval = 0)

    if ero > 1:
        iso_seed = ndimage.grey_erosion(iso_seed, size=(ero,ero,ero), mode='constant', cval = 0)

    labels,num = ndimage.label(iso_seed, structure=ndimage.generate_binary_structure(img.ndim, 1))
   # print(num,'components')
    regions = ndimage.find_objects(labels)

    if num > 1:
        hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
        argsort_hist = np.argsort(hist)[::-1]
        i = argsort_hist[0]
        r = regions[i]
        mask = labels[r] == i+1
        box = iso_seed[r].copy()
        box[~mask] = 0
        iso_seed = box.copy()

    gauss = ndimage.gaussian_filter(iso_seed, sigma=sigma, mode='constant', cval=0, truncate=4)
    iso_seed[gauss < 0.75*thr] = 0

    if w:
        outname = '{}{}_p{}_d{}_t{}_o{}_e{}_g{}.tif'.format(dst,bname,padding, dil, thr, op, ero, sigma)
        tf.imwrite(outname, iso_seed, photometric='minisblack', compress=3)


    return iso_seed

def refine_pesky_seeds(dst, img, cutoff=1e-2, opening=7, write_tif=False, median=1e5, med_range=2000, bname='test'):
    split_further = False
    tol = 1.5
    img = ndimage.grey_opening(img, size=(opening, opening, opening), mode='constant', cval = 0)
    sizes = []
    locs = []

    counter = 0
    labels,num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    print(num,'components')

    regions = ndimage.find_objects(labels)
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    sz_hist = ndimage.sum(hist)
    argsort_hist = np.argsort(hist)[::-1]
    print('hist', hist[argsort_hist])
    print('size =',sz_hist)

    if num > 1:

        for j in range(len(regions)):
            i = argsort_hist[j]
            r = regions[i]
            if (hist[i]/sz_hist > cutoff) and (math.fabs(hist[i] - median) < tol*med_range):
                z0,y0,x0,z1,y1,x1 = r[0].start,r[1].start,r[2].start,r[0].stop,r[1].stop,r[2].stop
                mask = labels[r]==i+1
                box = img[r].copy()
                box[~mask] = 0
                print('{}\t(x,y,z)=({},{},{}),\t (w,h,d)=({},{},{})'.format(j,x0,y0,z0,box.shape[2],box.shape[1],box.shape[0]))
                sizes.append((box.shape[2],box.shape[1],box.shape[0]))
                locs.append((x0,y0,z0))
                if write_tif:
                    tf.imwrite('{}{}_{}.tif'.format(dst,bname,counter),box,photometric='minisblack',compress=3)
                counter += 1

                print('---\n')

            elif math.fabs(hist[i] - median) >= tol*med_range:
                print('seed', bname,'_',j,' is too large/small.')
                split_further = True


    else:
        print(bname, 'could not be broken up further.')
        split_further = True

    return locs, sizes, split_further

def open_decompose(dst, box, cutoff=1e-2, opening = 7, write_tif=False, bname='test'):
    sizes = []
    locs = []
    box = ndimage.grey_opening(box, size=(opening, opening, opening), mode='constant', cval = 0)

    olabels,onum = ndimage.label(box, structure=ndimage.generate_binary_structure(box.ndim, 1))

    oregions = ndimage.find_objects(olabels)
    ohist,obins = np.histogram(olabels, bins=onum, range=(1,onum+1))
    osz_hist = ndimage.sum(ohist)
    oargsort_hist = np.argsort(ohist)[::-1]
    for k in range(len(oregions)):
        l = oargsort_hist[k]
        r = oregions[l]
        if(ohist[l]/osz_hist > cutoff):
            z0,y0,x0,z1,y1,x1 = r[0].start,r[1].start,r[2].start,r[0].stop,r[1].stop,r[2].stop
            omask = olabels[r]==l+1
            obox = box[r].copy()
            obox[~omask] = 0
            sizes.append((obox.shape[2],obox.shape[1],obox.shape[0]))
            locs.append((x0,y0,z0))
            if write_tif:
                tf.imwrite('{}{}_comp_{}.tif'.format(dst,bname,k),obox,photometric='minisblack',compress=3)

    return locs, sizes, ohist[oargsort_hist]

def preliminary_extract(src, seeddst, figdst, fname, bname, cutoff=1e-2, threshold = 200, write_tif=False, med_tol=0.5, op=7):

    img = tf.imread(src+fname)

    sname = os.path.splitext(fname)[0]
    sname = '_'.join(sname.split('_')[0:3])

    img[img < threshold] = 0
    locs, sizes, hist = open_decompose(seeddst, img, bname=bname, write_tif=write_tif, opening=op)

    diff = np.abs(np.ediff1d(hist))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    med = np.median(hist)
    med_cut = med_tol*med

    axes[0].axhline(med, c='m', lw=3, label='median')
    axes[0].axhline(med+med_cut, c='m', lw=3, ls=':', label='median + {:.0f}%'.format(100*med_tol))
    axes[0].axhline(med-med_cut, c='m', lw=3, ls=':')

    axes[0].plot(hist, color='blue', marker='o', lw=0, ms=9)
    axes[0].set_xlabel('seed', fontsize=18)
    axes[0].set_ylabel('volume', fontsize=18)
    axes[0].legend(fontsize=15)

    axes[1].axhline(med_cut, c='m', lw=3, ls=':', label='{:.0f}% median'.format(100*med_tol))
    axes[1].plot(diff, color='red', marker='o', lw=0, ms=9)
    axes[1].set_xlabel('seed', fontsize=18)
    axes[1].set_ylabel('| change in size |', fontsize=18)
    axes[1].legend(fontsize=15)
    fig.suptitle(sname, fontsize=24);

    fig.savefig(figdst + sname + '_preliminary.jpg', dpi=100, format='jpg', pil_kwargs={'optimize':True})

    to_split = []
    to_ignore = []

    for i in range(len(diff)):
        if diff[i] > med_cut  and hist[i] - med > med_cut :
            to_split.append(i+1)
        if diff[i] > med_cut  and med - hist[i+1] > med_cut :
            to_ignore.append(i+1)

    print(len(to_split), to_split)
    print(len(to_ignore), to_ignore)

    if len(to_split) == 0:
        start_seed = 0
    else:
        start_seed = to_split[-1]

    if len(to_ignore) == 0:
        end_seed = len(locs)
    else:
        end_seed = to_ignore[0]

    return locs, sizes, med, med_cut, start_seed, end_seed

def extract_refinement(dst, bname, med, med_cut, start_seed, end_seed, threshold=200, op=7, iter_tol=3, w=False):
    locs0 = []
    sizes0 = []
    to_ignore = []

    if start_seed > 0:
        for i in range(start_seed):
            iter0 = 0
            threshold0 = threshold + 5
            opening0 = op
            split_further = True

            l_name = bname + '_comp_{}.tif'.format(i)
            l_seed = tf.imread(dst + l_name)

            while ((split_further == True) and (iter0 < iter_tol)):
                print(i, '\tthreshold:',threshold0, '\topening:', opening0)
                l_seed[ l_seed < threshold0 ] = 0
                locs_temp, sizes_temp,split_further = refine_pesky_seeds(dst, l_seed, cutoff=1e-2, opening=opening0,
                                                                         write_tif=w, median=med,
                                                                         med_range=med_cut,
                                                                         bname=os.path.splitext(l_name)[0])
                iter0 += 1
                if iter0 % 2 == 0:
                    threshold0 +=2
                else:
                    opening0 +=1

                print('#################')

            if split_further:
                to_ignore.append(i)

            locs0.append(locs_temp)
            sizes0.append(sizes_temp)

    return locs0, sizes0, to_ignore

def seed_reconstruction(src, dst, fname, locs, sizes, locs0, sizes0, start_seed, end_seed, to_ignore,
                        padding=7, dil=4, thr=120, op=7, ero=1, sigma= 3, write_file=False):

    img = tf.imread(src+fname)
    bname = os.path.splitext(fname)[0]
    bname = '_'.join(bname.split('_')[1:3])

    if start_seed > 0:
        for i in range(start_seed):
            if i not in to_ignore:
                seed_files = sorted(glob.glob(dst + bname + '_comp_{}_*'.format(i)))
                for j in range(len(seed_files)):
                    seed = tf.imread(seed_files[j])
                    seed_loc = (locs[i][0] + locs0[i][j][0], locs[i][1] + locs0[i][j][1], locs[i][2] + locs0[i][j][2])
                    seed_size = sizes0[i][j]
                    seed_name = 'seed_{}_{}'.format(i,j)
                    iso_seed = seed_template(dst, img, seed, seed_loc,seed_size, padding=padding, op=op, dil=dil,
                                             ero=ero, thr=thr, sigma=sigma, bname=seed_name, w=write_file)

    for i in range(start_seed, end_seed, 1):
        seed = tf.imread(dst + bname + '_comp_{}_0.tif'.format(i))
        pos , size = locs[i], sizes[i]
        seed_name = 'seed_{}_0'.format(i)
        iso_seed = seed_template(dst, img, seed, pos, size, padding=padding, op=op, dil=dil,
                                 ero=ero, thr=thr, sigma=sigma, bname=seed_name, w=write_file)

    if end_seed < len(locs):
        for i in range(end_seed, len(locs)):
            seed = tf.imread(dst + bname + '_comp_{}_0.tif'.format(i))
            pos , size = locs[i], sizes[i]
            seed_name = 'seed_{}_0'.format(i)
            iso_seed = seed_template(dst, img, seed, pos, size, padding=padding, op=op, dil=dil+2,
                                     ero=ero, thr=thr, sigma=sigma, bname=seed_name, w=write_file)

def seed_isolation(src, figdst, fname, cutoff=1e-2, threshold = 200, med_tol=0.5,
                   padding=7, dil=4, thr=120, op=7, ero=1, sigma= 3, iter_tol=3, write_file=True):

    bname = os.path.splitext(fname)[0]
    bname = '_'.join(bname.split('_')[1:3])
    dst = src + bname + '_seeds/'

    if os.path.isdir(dst):
        pass
    else:
        os.makedirs(dst)
        print('directory', bname + '_seeds created')

    locs,sizes,med,med_cut,start_seed,end_seed = preliminary_extract(src,dst,figdst,fname,bname, threshold = threshold,
                                                                     write_tif=write_file,med_tol=med_tol, op=op)

    print('\n#####################\n')
    print('median size = ', med, '. 50% = ', med_cut)
    print('[',med+med_cut,'<->',med-med_cut,']')
    print('start = ', start_seed,'; end = ', end_seed)
    print('\n^^^^^^^^^^^^^^\n')

    locs0, sizes0, to_ignore = extract_refinement(dst, bname, med, med_cut, start_seed, end_seed,
                                       threshold=threshold, op=op, iter_tol=iter_tol, w=write_file)

    for i in range(start_seed, len(locs), 1):
        seed_file0 = dst + bname + '_comp_{}.tif'.format(i)
        seed_file  = dst + bname + '_comp_{}_0.tif'.format(i)
        os.rename(seed_file0, seed_file)

    seed_reconstruction(src, dst, fname, locs, sizes, locs0, sizes0, start_seed, end_seed, to_ignore,
                        padding=padding, dil=dil, thr=thr, op=op, ero=ero, sigma=sigma,
                        write_file=write_file)

    return locs, sizes, start_seed, end_seed, to_ignore

def spike_hists(src):
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))

    barley_files = sorted(glob.glob(src + '*.tif'))
    for i in range(4):
        img = tf.imread(barley_files[i])
        fname = os.path.basename(barley_files[i])
        bname = os.path.splitext(fname)[0]
        bname = '_'.join(bname.split('_')[1:3])
        hist,bins = np.histogram(img,bins=2**(img.dtype.itemsize*8),range=(0,2**(img.dtype.itemsize*8)))
        hist[0] = 0
        axes[i//2, i%2].plot(np.arange(2**(img.dtype.itemsize*8)), np.log(hist+1))
        axes[i//2, i%2].set_title(bname)

def missing_summary(scanId, color, locs, sizes, to_ignore):
    missing_seed = np.zeros((len(to_ignore), 4), np.float32)

    for i in range(len(to_ignore)):
        missing_seed[i, :] = [to_ignore[i],
                              locs[to_ignore[i]][0] + 0.5*sizes[to_ignore[i]][0],
                              locs[to_ignore[i]][1] + 0.5*sizes[to_ignore[i]][1],
                              locs[to_ignore[i]][2] + 0.5*sizes[to_ignore[i]][2]]

    bar = (pd.DataFrame([scanId, color])).T
    bar.columns = ['scan', 'color']
    bar = pd.concat([bar]*len(to_ignore))
    bar.index = list(range(len(to_ignore)))

    df = pd.DataFrame(missing_seed, columns=['seedNo', 'centerX', 'centerY', 'centerZ'])

    foo = pd.merge(bar,df, left_index=True, right_index=True).astype({'seedNo': 'uint8'})

    return foo

def seed_brew(src,dst,figdst,cutoff,threshold,med_tol,padding,dil,thr,op,ero,sigma,write_file):
    barley_files = sorted(glob.glob(src + '*.tif'))
    missing_seeds = pd.DataFrame()
    scanId = os.path.normpath(src).split(os.path.sep)[-1]

    for i in range(len(barley_files)-1):
        fname = os.path.basename(barley_files[i])
        color = (os.path.splitext(fname)[0]).split('_')[2]

        locs, sizes,ss,ee,to_ignore = seed_isolation(src, figdst,fname, cutoff, threshold, med_tol,
                                                     padding, dil, thr, op, ero, sigma, write_file)
        print("TO IGNORE", to_ignore)
        if len(to_ignore) > 0:
            df = missing_summary(scanId, color, locs, sizes, to_ignore)
            missing_seeds = missing_seeds.append(df, ignore_index=True)

    if not missing_seeds.empty:
        missing_seeds.to_csv(dst + scanId + '_ignore.csv', index=False)


def tiff2coords(img, center=True):
    coords = np.nonzero(img)
    coords = np.vstack(coords).T
    if center:
        origin = -1*np.mean(coords, axis=0)
        coords = np.add(coords, origin)

    return coords

def dummy_dict(dictionary):
    foo = dict()
    for fname in dictionary:
        foo[fname] = 'foo'
    return foo

def read_boxes(src):
    barley_files = sorted(glob.glob(src + '*.tif'))
    spikes = len(barley_files)

    if spikes > 5 or spikes < 2:
        print('Found',spikes,'connected components. Expected between 2 and 5. Aborting.')
        sys.exit(0)

    img = dict()
    for i in range(spikes):
        fname = os.path.split(barley_files[i])[1]
        img[fname] = tf.imread(barley_files[i])

    boxes = dict()
    for fname in img:
        coords = []
        fields = os.path.splitext(fname)[0].split('_')
        for f in fields:
            if f[0] in 'lxyz':
                coords.append(f[1:])
        coords = np.array(coords, dtype='int')
        d,h,w = img[fname].shape
        boxes[fname] = (coords[1], coords[2], coords[3], w,h,d, coords[0])

    marker = os.path.split(barley_files[-1])[1]

    return img, boxes, marker

def find_centers(img, boxes, slices=10):
    dcoords = dict()
    acoords = np.empty((len(boxes), 2), dtype=np.float64, order='C')
    for idx,fname in enumerate(boxes):
        foo = np.round(np.mean(tiff2coords(img[fname][slices,:,:], center=False), axis=0))
        foo = np.add(foo, np.array([boxes[fname][1], boxes[fname][0]]))
        acoords[idx, :] = foo
        dcoords[fname] = foo

    return acoords, dcoords

def euclidean_dists(centers, marker):
    dummy = dummy_dict(centers)
    m_coord = centers[marker]
    euclidean = dict()
    idx = 0
    del dummy[marker]

    for fname in dummy:
        euclidean[fname] = np.sqrt(np.sum((m_coord - centers[fname])**2))
        #print(euclidean[fname])

    return euclidean


def coloring_spikes(dcenters, euclidean, hull, col_name = ['Red', 'Green', 'Orange', 'Blue']):
    colors = dict()
    red = min(euclidean, key=lambda key: euclidean[key])
    colors[red] = col_name[0]

    foo = hull.points[hull.vertices]

    idx_r = 0
    for i in range(foo.shape[0]):
        if np.sum(dcenters[red] == foo[i]) == foo.shape[1]:
            idx_r = i

    for i in range(1,foo.shape[0]):
        for fname,xy in dcenters.items():
            if np.sum(xy == foo[(idx_r+i)%foo.shape[0]]) == foo.shape[1]:
                colors[fname] = col_name[i]

    return colors

def render_alignment(dst, sname, boxes, colors, write_fig=False):
    fig = plt.figure(figsize=(10,6))
    fig.suptitle(sname, fontsize=25)
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=80., azim=40)

    for fname in boxes:
        x,y,z,w,h,d,l = boxes[fname]
        cx,cy = x+w/2,y+h/2
        ax.plot([x,x,x,x,x+w,x+w,x+w,x+w,x],[y,y,y+h,y+h,y+h,y+h,y,y,y],[z,z+d,z+d,z,z,z+d,z+d,z,z],c=colors[fname].lower())
        ax.text(x+w/2,y+h/2,z+d,'{} {}'.format('L', l),None, color=colors[fname].lower(), ha='center', va='center')

    if write_fig:
        plt.savefig(dst+sname+'_alignment.jpg', dpi=150, format='jpg', pil_kwargs={'optimize':True})

def barley_labeling(src, dst, rename=True):
    img,boxes,marker = read_boxes(src)

    centers,dcenters = find_centers(img, boxes)
    hull = spatial.ConvexHull(centers[:-1,:])

    if len(hull.vertices) != 4:
        print(src,'\nConvex Hull reports less than 4 spikes!!!.')
        sys.exit(0)

    euclidean = euclidean_dists(dcenters, marker)
    colors = coloring_spikes(dcenters, euclidean, hull)
    colors[marker] = 'Black'

    sname = marker.split('_')[0]
    render_alignment(dst, sname, boxes,colors)
    braces = '{}_'

    if rename:
        for fname in colors:
            splt = fname.split('_')
            cname = braces*len(splt) + '{}'
            cname = cname.format(sname,splt[1],colors[fname],*(splt[2:]))
            os.rename(src + fname, src + cname)

    return dst, sname, boxes, colors


def barley_brew(dst, tiff_file, basis, sigma=3, thr1=55, ero=(7,7,7), dil=(5,5,5), thr2=30, op=(1,11,11),
                cutoff = 1e-2, flex=2):
    src, fname = os.path.split(tiff_file)
    src = src + '/'
    bname = os.path.splitext(fname)[0]
    img = tf.imread(tiff_file)
    adjust_by = np.loadtxt(dst+'normalize/' + bname+'_yvals.csv', dtype='float', delimiter=',')

    if fname != basis:
        img = normalize_density(img, adjust_by)

    img = misc_cleaning(img, sigma=3, thr1=thr1, ero=ero, dil=dil, thr2=thr2, op=op)
    dst = dst + 'spikes/' + bname + '/'
    if not os.path.isdir(dst):
        os.makedirs(dst)
        print('directory', dst, 'created')
    separate_pruned_spikes(dst, bname, img, cutoff=cutoff, flex=flex)

    return src, dst, fname, bname

def load_metadata(meta,scan_name,bcolor,seed_no,spike_label,columns=[]):

    scan_meta = meta[meta['Scan'] == scan_name]
    spike_meta = scan_meta[scan_meta['Color'] == bcolor]

    spike_meta = pd.concat([spike_meta]*seed_no)

    spike_meta['Label'] = spike_label

    if len(columns) > 0:
        for col in columns:
            spike_meta[col] = 1.0

    return spike_meta

def find_tip(coords, x,y,z):
    maxes = np.max(coords, axis=0)
    max_vox = coords[coords[:, z] == maxes[z]]
    if len(max_vox) > 1 :
        maxesz = np.max(max_vox, axis=0)
        max_vox = max_vox[max_vox[:, y] == maxesz[y]]

        if len(max_vox) > 1:
            maxesy = np.max(max_vox, axis=0)
            max_vox = max_vox[max_vox[:, x] == maxesy[x]]

    return np.squeeze(max_vox)

def rotateSVD(coords, max_vox, x=0,y=1,z=2):
    u, s, vh = np.linalg.svd(coords, full_matrices=False)
    sigma = np.sqrt(s)
    seed = np.matmul(coords, np.transpose(vh))
    y_pos = seed[seed[:,y] > 0]
    y_neg = seed[seed[:,y] < 0]

    y_posmax = np.max(y_pos, axis=0)
    y_posmin = np.min(y_pos, axis=0)
    y_negmax = np.max(y_neg, axis=0)
    y_negmin = np.min(y_neg, axis=0)
    hzp = np.squeeze(y_pos[y_pos[:,z]==y_posmax[z]])[y] - np.squeeze(y_neg[y_neg[:,z]==y_negmax[z]])[y]
    hzn = np.squeeze(y_pos[y_pos[:,z]==y_posmin[z]])[y] - np.squeeze(y_neg[y_neg[:,z]==y_negmin[z]])[y]

    rotZ = False
    if hzn > hzp:
        seed[:, z] = -1.0*seed[:, z]
        rotZ = True

    rotX = False
    if max_vox[0] < 0:
        seed[:, x] = -1.0*seed[:, x]
        rotX = True

    return seed, sigma, vh, rotZ, rotX

def seed_lengths(seed, x=0,y=1,z=2):
    maxs, mins = np.max(seed, axis=0), np.min(seed, axis=0)
    length, width, heightmax = maxs - mins
    height = np.max(seed[ np.abs(seed[:,y]) < 0.5 ], axis=0)[z] - mins[z]

    return length, width, height, heightmax


def seed_area_vol(img, coords, border):
    surface = ndimage.convolve(img, border, np.int8, 'constant', cval=0)

    surface[ surface < 0 ] = 0
    area_sq = np.sum(surface)
    area_cube = np.sum(surface > 0)
    vol = np.sum(img)

    hull = spatial.ConvexHull(coords)

    area_ratio = area_sq/hull.area
    vol_ratio = vol/hull.volume

    return area_sq,area_cube,vol,hull.area,hull.volume,area_ratio,vol_ratio

def save_alignment(dst, bname, seed, sigma, algn_max_vox):
    np.savetxt(dst+bname+'_sigma.csv', sigma, fmt='%.5e', delimiter = ',')
    np.savetxt(dst+bname+'_coords.csv', seed, fmt='%.5e', delimiter = ',')

def traditional_summary(dst, csv_path, scan_path, color_list, border_mask, save_coords=False):

    scan_name = os.path.normpath(scan_path).split(os.path.sep)[-1]
    x,y,z = 2,1,0
    traits = ['Length', 'Width', 'Height', 'HeightMax', 'Shell', 'Area', 'Vol', 'ConvexArea', 'ConvexVol', 'ConvexAreaRatio', 'ConvexVolRatio']

    meta = pd.read_csv(csv_path)

    for color in color_list:
        print('***********************************')
        print(scan_name, color)
        seed_files = glob.glob(scan_path + '*' + color + '*/seed_*_p*.tif')
        Tag, Length,Width,Height,HeightMax,Shell,Area,Vol,ConvexArea,ConvexVol,ConvexAreaRatio,ConvexVolRatio = [],[],[],[],[],[],[],[],[],[],[],[]

        if len(seed_files) < 1:
            print('Seeds not found in ', scan_path)

        else:

            csvdst = dst + scan_name + '/'
            if not os.path.isdir(csvdst):
                os.makedirs(csvdst)

            foodst = csvdst + color + '/'
            if save_coords and not os.path.isdir(foodst):
                os.makedirs(foodst)

            src, fname = os.path.split(seed_files[0])
            label = ((os.path.normpath(src).split(os.path.sep)[-1]).split('_')[0])[-1]

            summary = load_metadata(meta,scan_name,color, len(seed_files), label, traits)

            for seed_file in seed_files:

                raw, fname = os.path.split(seed_file)
                bname = '_'.join(os.path.splitext(fname)[0].split('_')[:3])

                img = tf.imread(seed_file)

                coords = tiff2coords(img, center=True)

                area_sq,area_cube,vol,hullarea,hullvol,area_ratio,vol_ratio = seed_area_vol(img, coords, border_mask)


                max_vox = find_tip(coords, x,y,z)

                seed, sigma, vh, algn_max_vox, rotZ, rotX = rotateSVD(coords, max_vox)
                length, width, height, heightmax = seed_lengths(seed)

                Tag.append(bname)
                Length.append(length)
                Width.append(width)
                Height.append(height)
                HeightMax.append(heightmax)
                Area.append(area_sq)
                Shell.append(area_cube)
                Vol.append(vol)
                ConvexArea.append(hullarea)
                ConvexVol.append(hullvol)
                ConvexAreaRatio.append(area_ratio)
                ConvexVolRatio.append(vol_ratio)

                if save_coords:
                    save_alignment(foodst, bname, seed, sigma, algn_max_vox)


            summary['Tag'] = Tag
            summary['Length'] = Length
            summary['Width'] = Width
            summary['Height'] = Height
            summary['HeightMax'] = HeightMax
            summary['Shell'] = Shell
            summary['Area'] = Area
            summary['Vol'] = Vol
            summary['ConvexArea'] = ConvexArea
            summary['ConvexVol'] = ConvexVol
            summary['ConvexAreaRatio'] = ConvexAreaRatio
            summary['ConvexVolRatio'] = ConvexVolRatio

            summary.to_csv(csvdst + scan_name + '_' + color +'_summary.csv', index=False)

    return summary

def pole_directions(parallels, meridians, x=0, y=1, z=2, tol=1e-10):
    dirs = np.zeros((2*(meridians*parallels)-meridians+2, 3), dtype=np.float64)
    idx = 1

    dirs[0, :] = np.array([0,0,0])
    dirs[0, z] = 1

    for i in range(parallels):
        theta = (i+1)*math.pi/(2*parallels)
        for j in range(meridians):
            phi = j*math.tau/meridians
            dirs[idx,x] = math.cos(phi)*math.sin(theta)
            dirs[idx,y] = math.sin(phi)*math.sin(theta)
            dirs[idx,z] = math.cos(theta)
            idx += 1

    for i in range(parallels-1):
        theta = (i+1)*math.pi/(2*parallels) + 0.5*math.pi
        for j in range(meridians):
            phi = j*math.tau/meridians
            dirs[idx,x] = math.cos(phi)*math.sin(theta)
            dirs[idx,y] = math.sin(phi)*math.sin(theta)
            dirs[idx,z] = math.cos(theta)
            idx += 1


    dirs[-1, :] = np.array([0,0,0])
    dirs[-1, z] = -1
    dirs[np.abs(dirs) < tol] = 0

    return dirs

def sum_to_coords(coord, sums):
    foo = np.array(coord)
    if len(coord) != len(sums):
        print('len(coord) != len(sums)')
    else:
        for i in range(len(coord)):
            foo[i] += sums[i]

    return tuple(foo)


def centerVertices(verts):
    origin = -1*np.mean(verts, axis=0)
    verts = np.add(verts, origin)
    return verts

def tiff2dict(img):
    coords = np.nonzero(img)
    coords = np.vstack(coords).T
    keys = [tuple(coords[i,:]) for i in range(len(coords))]
    dcoords = dict(zip(keys, range(len(coords))))
    return dcoords

def ECC(verts, cells, filtration, T=32, bbox=None):

    if bbox is None:
        minh = np.min(filtration)
        maxh = np.max(filtration)
    else:
        minh,maxh = bbox

    buckets = [None for i in range(len(cells))]

    buckets[0], bins = np.histogram(filtration, bins=T, range=(minh, maxh))

    for i in range(1,len(buckets)):
        buckets[i], bins = np.histogram(np.max(filtration[cells[i]], axis=1), bins=T, range=(minh, maxh))

    ecc = np.zeros_like(buckets[0])
    for i in range(len(buckets)):
        ecc = np.add(ecc, ((-1)**i)*buckets[i])

    return np.cumsum(ecc)

def ECT(verts, cells, directions,T=32, bbox=None):

    ect = np.zeros(T*directions.shape[0], dtype=int)

    for i in range(directions.shape[0]):
        heights = np.sum(verts*directions[i,:], axis=1)
        ect[i*T : (i+1)*T] = ECC(verts, cells, heights, T, bbox)

    return ect

def neighborhood_setup(dimension):
    neighs = sorted(list(itertools.product(range(2), repeat=dimension)), key=np.sum)[1:]
    subtuples = dict()
    for i in range(len(neighs)):
        subtup = [0]
        for j in range(len(neighs)):
            if np.all(np.subtract(neighs[i], neighs[j]) > -1):
                subtup.append(j+1)
        subtuples[neighs[i]] = subtup

    return neighs, subtuples

def neighborhood(voxel, neighs, hood, dcoords):
    hood[0] = dcoords[voxel]
    neighbors = np.add(voxel, neighs)
    for j in range(1,len(hood)):
        key = tuple(neighbors[j-1,:])
        if key in dcoords:
            hood[j] = dcoords[key]
    return hood

def complexify(img, center=True):
    coords = np.nonzero(img)
    coords = np.vstack(coords).T
    keys = [tuple(coords[i,:]) for i in range(len(coords))]
    dcoords = dict(zip(keys, range(len(coords))))
    neighs, subtuples = neighborhood_setup(img.ndim)
    binom = [special.comb(img.ndim, k, exact=True) for k in range(img.ndim+1)]

    hood = np.zeros(len(neighs)+1, dtype=int)-1
    cells = [[] for k in range(img.ndim+1)]

    for voxel in dcoords:
        hood.fill(-1)
        hood = neighborhood(voxel, neighs, hood, dcoords)
        nhood = hood > -1
        if np.all(nhood):
            c = 0
            for k in range(1, img.ndim + 1):
                for j in range(binom[k]):
                    cell = hood[subtuples[neighs[c]]]
                    cells[k].append(cell)
                    c += 1
        else:
            c = 0
            for k in range(1, img.ndim):
                for j in range(binom[k]):
                    cell = nhood[subtuples[neighs[c]]]
                    if np.all(cell):
                        cells[k].append(hood[subtuples[neighs[c]]])
                    c += 1

    cells = [np.array(cells[k]) for k in range(len(cells))]
    if center:
        cells[0] = centerVertices(coords)
    else:
        cells[0] = coords

    return cells

def plot_pole_directions(directions, titleplot = 'title', parallels=8, meridians=12, save_fig=False, dst = './', filename = 'sphere'):

    pdirections = pole_directions(parallels,meridians,x=1,y=0,z=2)
    viridis = cm.get_cmap('viridis', parallels*2-1)
    opacity = np.linspace(0.25,0.9,parallels*2-1)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    axlen = np.linspace(np.min(directions), np.max(directions), 50)
    zeros = np.repeat(0,len(axlen))

    ax.plot(axlen, zeros, zeros, c='r', lw=4, alpha=0.5)
    ax.plot(zeros, axlen, zeros, c='b', lw=4, alpha=0.5)
    ax.plot(zeros, zeros, axlen, c='g', lw=4, alpha=0.5)
    ax.scatter(directions[:,0],directions[:,1],directions[:,2], marker='^', s=160, c='m')

    for i in range(parallels*2-1):
        ax.plot(pdirections[(i*meridians + 1):((i+1)*meridians + 1),0],
                pdirections[(i*meridians + 1):((i+1)*meridians + 1),1],
                pdirections[(i*meridians + 1):((i+1)*meridians + 1),2],
                c=viridis.colors[i,:3],
                alpha = opacity[i],
                lw = 2.5)
        ax.plot(pdirections[np.array((i*meridians + 1,(i+1)*meridians)),0],
                pdirections[np.array((i*meridians + 1,(i+1)*meridians)),1],
                pdirections[np.array((i*meridians + 1,(i+1)*meridians)),2],
                c=viridis.colors[i,:3],
                alpha = opacity[i],
                lw = 2.5)

    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_axis_off()
    #ax.set_xlabel('X Axis', fontsize=24)
    #ax.set_ylabel('Y Axis', fontsize=24)
    #ax.set_zlabel('Z Axis', fontsize=24)
    ax.set_title(titleplot,fontsize=28, pad=0, y=0.83);
    plt.tight_layout();

    if save_fig:
        plt.savefig(dst+filename+'.jpg', bbox_inches='tight',
                    dpi=72, format='jpg', pil_kwargs={'optimize':True});
        plt.savefig(dst+filename+'.pdf', dpi=72,
                    bbox_inches='tight', format='pdf');
        plt.close()

def random_directions(N=50, r=1):
    rng = np.random.default_rng()
    z = rng.uniform(-r, r, N)
    phi = rng.uniform(0, 2*np.pi, N)
    x = np.sqrt(r**2 - z**2)*np.cos(phi)
    y = np.sqrt(r**2 - z**2)*np.sin(phi)

    return np.column_stack((x,y,z))

def regular_directions(N=50, r=1):
    dirs = np.zeros((N, 3), dtype=np.float64)
    i = 0
    a = 4*np.pi*r**2/N
    d = np.sqrt(a)
    Mtheta = np.round(np.pi/d)
    dtheta = np.pi/Mtheta
    dphi = a/dtheta
    for m in range(int(Mtheta)):
        theta = np.pi*(m + 0.5)/Mtheta
        Mphi = np.round(2*np.pi*np.sin(theta)/dphi)
        for n in range(int(Mphi)):
            phi = 2*np.pi*n/Mphi

            dirs[i,:] = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
            i += 1

    return dirs

def plot_3Dprojections(seed, title='title', markersize=2, writefig=False, dst='./', dpi=150):
    axes = ['X','Y','Z']
    fig, ax = plt.subplots(1,3,figsize=(12,4))

    for i in range(3):
        proj = []
        for j in range(3):
            if j != i:
                proj.append(j)
        ax[i].plot(seed[:,proj[0]], seed[:,proj[1]], '.', ms=markersize, c='y')
        ax[i].set_xlabel(axes[proj[0]])
        ax[i].set_ylabel(axes[proj[1]])
        ax[i].set_title(axes[i] + ' Projection')
        ax[i].set_aspect('equal');

    fig.suptitle(title, y=0.95, fontsize=20)
    plt.tight_layout();

    if writefig:
        filename = '_'.join(title.split(' ')).lower()
        plt.savefig(dst + filename + '.png', dpi=dpi, format='png', bbox_inches='tight',
                    facecolor='white', transparent=False)
        plt.close();

def img_align(img, x=2, y=1, z=0):

    coords = tiff2coords(img, 0)
    max_vox = find_tip(coords, x,y,z)
    seed, sigma, vh, rotZ, rotX = rotateSVD(coords, max_vox)

    return seed

def radial_tracing(data, granularity=100, resolution=1e-3):
    angle = 0
    delta = math.pi/granularity
    contour = np.zeros((2*granularity,data.ndim))
    a,b = (np.max(data, axis=0) - np.min(data, axis=0))*0.5

    for i in range(granularity):
        angle += delta
        direction = np.around(np.array([a*math.cos(angle), b*math.sin(angle)]), decimals=10)
        direction = direction / np.sqrt(np.sum(direction**2))
        heights = np.sum(data*direction, axis=1)
        norms = np.sqrt(np.sum(data*data, axis=1)) + 1e-6
        cos_angle = np.abs(heights/norms)
        lighthouse = np.where(1-cos_angle < resolution, heights, 0)
        maxi,mini = np.max(lighthouse), np.min(lighthouse)

        contour[i,:] = np.array([maxi * direction[0], maxi*direction[1]])
        contour[granularity+i:] = np.array([mini * direction[0], mini*direction[1]])

    return contour

def subsample_landmarks(roots, tol=1e-2):

    anchor = roots[0]
    sample = []
    midway = [anchor]

    for i in range(len(roots)-1):

        anchor = midway[0]
        if roots[i+1] - anchor > tol:
            sample.append(np.mean(np.array(midway)))
            midway = [roots[i+1]]

        else:
            midway.append(roots[i+1])

    sample.append(roots[-1])

    return np.array(sample)

def critical_spline(t, dspl, depth=3):

    root = [[None for j in range(depth)] , [None for j in range(depth)]]

    for axis in range(len(root)):
        spli = interpolate.InterpolatedUnivariateSpline(t, dspl[axis])
        root[axis][0] = spli.roots()

        spli = interpolate.InterpolatedUnivariateSpline(t, spli.derivative().__call__(t))
        root[axis][1] = spli.roots()

        spli = interpolate.InterpolatedUnivariateSpline(t, spli.derivative().__call__(t))
        root[axis][2] = spli.roots()

    return root

def scaling_array(data, axis=0, ddof=1):
    scaled_lm = data - np.mean(data, axis=axis)
    scaled_lm = scaled_lm/np.std(data, axis=axis, ddof=ddof)
    return scaled_lm
