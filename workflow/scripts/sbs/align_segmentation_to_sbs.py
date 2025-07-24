from tifffile import imread, imwrite
<<<<<<< HEAD

from lib.sbs.align_cycles import align_seg_to_sbs

# load segmentation stack
seg_stack = imread(snakemake.input[0])

# load aligned SBS stack and use DAPI from first cycle as reference
aligned_sbs = imread(snakemake.input[1])
ref_dapi = aligned_sbs[0, 0]

aligned_seg = align_seg_to_sbs(
    seg_stack,
    ref_dapi,
    dapi_index=snakemake.params.dapi_index,
=======
from lib.sbs.align_cycles import align_seg_to_sbs

# Load segmentation stack
seg_stack = imread(snakemake.input[0])

# Load aligned SBS data and extract reference DAPI channel
aligned_sbs = imread(snakemake.input[1])
dapi_cycle = int(snakemake.params.dapi_cycle)
dapi_index = snakemake.params.dapi_index
ref_dapi = aligned_sbs[dapi_cycle - 1, dapi_index]

# Align segmentation stack to SBS reference
aligned_seg = align_seg_to_sbs(
    seg_stack,
    ref_dapi,
    dapi_index=dapi_index,
>>>>>>> mgodzik/add_segmentation
    upsample_factor=snakemake.params.upsample_factor,
    window=snakemake.params.window,
)

imwrite(snakemake.output[0], aligned_seg)
