from pathlib import Path
from PIL import Image
from tqdm import tqdm


def format_repeated_slices(dataset_dir, output_dir, num_samples):
    for id, image in tqdm(enumerate(dataset_dir.iterdir(), start=1)):
        # Open image from 50k dataset & convert to grayscale
        im = Image.open(image).convert('L')
        # Create folder structure same as LIDC dataset
        img_out_dir = output_dir / f'patient_{int(id)}_nod_1_diag_1'
        img_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the image as tif
        for slice_num in range(8, 24):
            im.save(img_out_dir / f'slice_{int(slice_num)}.tif', 'TIFF')
        # Stop when required data samples have been generated
        if id >= num_samples:
            break


def format_unique_slices(dataset_dir, output_dir, num_samples):
    sample_id = 1
    slice_num = 8
    for image in tqdm(dataset_dir.iterdir()):
        # Open image from 50k dataset & convert to grayscale
        im = Image.open(image).convert('L')

        # For a new sample create it's folder
        if slice_num == 8:
            # Create folder structure same as LIDC dataset
            img_out_dir = output_dir / f'patient_{int(sample_id)}_nod_1_diag_1'
            img_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the image as tif
        im.save(img_out_dir / f'slice_{int(slice_num)}.tif', 'TIFF')

        # Update slice number
        if slice_num >= 23:
            slice_num = 8
            sample_id += 1
        else:
            slice_num += 1

        # Stop when required data samples have been generated
        if sample_id >= num_samples+1:
            break


if __name__ == '__main__':
    dataset_dir = Path('F:\\master\\random_data\\50K')
    num_samples = 1000

    output_dir = Path('F:\\master\\random_data\\50K_sample_1k_repeated_slices')
    format_repeated_slices(dataset_dir, output_dir, num_samples)

    output_dir = Path('F:\\master\\random_data\\50K_sample_1k_unique_slices')
    format_unique_slices(dataset_dir, output_dir, num_samples)

