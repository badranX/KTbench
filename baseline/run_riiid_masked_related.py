from baseline.dkt import bench_mask_label_dkt
from dkt import bench_dkt
from akt import bench_akt, bench_masked_akt

datasets = ['riiid2020']
bench_mask_label_dkt.main(datasets)
#bench_masked_akt.main(datasets1)
#round 2
bench_masked_akt.main(datasets)
#bench_masked_akt.main(datasets2)



