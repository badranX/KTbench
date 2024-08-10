#from baseline.akt import bench_akt_ml# bench_mask_label_akt
from baseline.dkt import bench_dkt, bench_dkt_ad, bench_fuse_dkt, bench_dkt_ml
from baseline.akt import bench_akt, bench_akt_ml, bench_akt_qm
from baseline.qikt import bench_qikt
from baseline.dkvmn import bench_dkvmn
from baseline.deep_irt import bench_deep_irt

datasets1 = ['assist2009', 'algebra2005']
datasets2 = ['duolingo2018_es_en', 'corr_assist2009', 'riiid2020']
datasets = datasets1 + datasets2

#DKT related
bench_dkt_ml.main(datasets)
bench_fuse_dkt.main(datasets)
bench_dkt_ad.main(datasets)
bench_dkt.main(datasets)

#AKT related
bench_akt_ml.main(datasets)
bench_akt_qm.main(datasets)
bench_akt.main(datasets)

#DKVMN, DeepIRT, QIKT
bench_qikt.main(datasets)
bench_dkvmn.main(datasets)
bench_deep_irt.main(datasets)