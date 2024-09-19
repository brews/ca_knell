############ What if we compress the ds #########
import os
from dask_gateway import GatewayCluster
import pandas as pd
import xarray as xr
import zarr

JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")

cluster = GatewayCluster(
    worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro"
)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(100)

cmip5 = xr.open_dataset(
    "gs://impactlab-data-scratch/brews/5515562a-9488-41d8-a655-2ae1ddd62f90/cmip5_concat.zarr",
    engine="zarr",
    chunks={},
)

# Might be easier to just parse all leaves of cmip5 but I feel this is likely
# to have a better error if something goes wrong.
idx_raw = []
ds_to_concat = []
for scenario_name, scenario_tree in cmip5.children.items():
    for source_name, source_tree in scenario_tree.children.items():
        k = f"{scenario_name}/{source_name}"
        print(k)  # DEBUG
        idx_raw.append(k)
        ds_to_concat.append(source_tree.to_dataset())
# TODO: Can we do this without importing and using Pandas (pd.Index)?
cmip5_ds = xr.concat(
    ds_to_concat,
    pd.Index(idx_raw, name="ensemble_member"),
    combine_attrs="drop_conflicts",
)
cmip5_ds = cmip5_ds.chunk({"time": 365, "lat": 360, "lon": 360})

compressor = zarr.Blosc(cname="zstd", clevel=9)
encoding = {}
for k, v in cmip5_ds.data_vars.items():
    encoding[k] = {
        "compressor": zarr.Blosc(cname="lz4", clevel=9, shuffle=1, blocksize=0)
    }
#    cmip5[k].encoding["compressor"] = zarr.Blosc(cname='lz4', clevel=9, shuffle=1, blocksize=0)
out_url = "gs://rhg-data-scratch/brews/764629be-e489-46fb-8800-8e7ad3e18cc2/cmip5_ds_lz4_clevel9.zarr"
cmip5_ds.to_zarr(out_url, mode="w", encoding=encoding)
print(out_url)
# Started at 2024-08-01T11:07
