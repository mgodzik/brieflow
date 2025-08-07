import pandas as pd

from lib.sbs.call_cells import call_cells

# load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# split reads into perturbation and UMI subsets based on the first base
perturb_reads = reads_data[reads_data.barcode.str.get(0).isin(["A", "G"])]
umi_reads = reads_data[reads_data.barcode.str.get(0).isin(["C", "T"])]

# load df_barcode_library
df_barcode_library = pd.read_csv(snakemake.params.df_barcode_library_fp, sep="\t")

# call cells
cells_data = call_cells(
    reads_data=perturb_reads,
    df_barcode_library=df_barcode_library,
    q_min=snakemake.params.q_min,
    barcode_col=snakemake.params.barcode_col,
    prefix_col=snakemake.params.prefix_col,
    df_UMI=None if umi_reads.empty else umi_reads,
    error_correct=snakemake.params.error_correct,
    sort_calls=snakemake.params.sort_calls,
)

# save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")
