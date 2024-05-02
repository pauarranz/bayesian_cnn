"""This code is a modification of the code present in the below repository:
    Apostolopoulos, I. (2020). LIDC-IDRI-Extract-64x64x16-nodules. Github. https://github.com/apjohndim/LIDC-IDRI-Extract-64x64x16-nodules
"""
import numpy as np
import pylidc as pl  # pip install pylidc
from PIL import Image
from pathlib import Path
import shutil


def nodules_extraction(data_path, out_path, slices_ext=(0, 1)):
    """

    :param slices_ext: set
        Defines the initial and final slice to extract. Being 0 the mean slice where the nodule is located.
            Example: if (-1, 1), 3 slices will be extracted (the mean one, the one before and the one after.
    :return:
    """
    # load all scans in object scan
    scans = pl.query(
        pl.Scan
    ).filter(
        pl.Scan.slice_thickness <= 5,
        pl.Scan.pixel_spacing <= 5
    )
    print(f'{scans.count()} scans have this characterisics: slice_thickness and pixel_spacing')

    for patient_path in data_path.iterdir():
        if patient_path.is_dir():
            pid = patient_path.name
            nodule_count = 0

            scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()  # obtain the scans

            print('[INFO] FOUND %4d SCANS' % len(scans))
            for scan in scans:  # scan object of this pid

                ann = scan.annotations
                vol = scan.to_volume()  # obtain the volume numpy array
                nods = scan.cluster_annotations()  # obtain the nodules object from this scan
                my_id = int(scan.patient_id.split('-')[-1])

                #print(len(scan.annotations))  # how many annotations
                print("'[INFO] %s has %d nodules." % (scan, len(nods)))

                it = len(nods)  # define how many nodules. nods is not iterable, be careful
                for nodule in range(0, it):
                    x = nods[nodule]  # get only the first annotation
                    x = x[0]  # grab the first annotation
                    #print(x.malignancy)
                    # Extract the list of slices in which the nodule appears
                    slices = x.contour_slice_indices
                    # Grab a relative coordinate
                    place = x.contours_matrix[0]
                    xc = place[0]
                    yc = place[1]
                    slide = slices[int(len(slices) / 2)]  # go to the mean slice
                    # for slide in slices:
                    nodule_count += 1

                    # Extract all requested slices
                    out_nod_path = out_path / f'patient_{my_id}_nod_{nodule_count}_diag_{x.malignancy}'
                    if not out_nod_path.is_dir():
                        out_nod_path.mkdir()
                    num_slices = len(list(range(slices_ext[0], slices_ext[1])))
                    print('[INFO] EXTRACTING %4d SLICES' % num_slices)
                    try:
                        for k in range(slices_ext[0], slices_ext[1]):  # I extract 16 slices
                            vol1 = vol[:, :, slide + k]  # get the image of the particular slice
                            vol1 = vol1[(xc - 32):(xc + 32), (yc - 32):(yc + 32)]  # for 64x64.
                            vol1 = (vol1 - np.amin(vol1)) / np.amax(vol1)  # adjust the pixel values to 0,1
                            vol1 = vol1 * 255  # adjust to 0,255
                            im = Image.fromarray(vol1)
                            im = im.convert("L")
                            # Save images
                            img_path = out_nod_path / f'slice_{k+num_slices}.tif'
                            im.save(str(img_path))
                    except:
                        shutil.rmtree(out_nod_path)
            my_id += 1


if __name__ == '__main__':
    data_path = Path('F:\\master\\manifest-1600709154662\\LIDC-IDRI\\')
    out_path = Path('F:\\master\\manifest-1600709154662\\nodules_16slices\\')
    nodules_extraction(data_path=data_path, out_path=out_path, slices_ext=(-8, 8))
