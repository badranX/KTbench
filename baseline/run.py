from baseline.akt import bench_mask_label_akt
from dkt import bench_dkt, bench_mask_label_dkt, bench_dkt_fuse
from akt import bench_akt, bench_mask_label_akt, bench_question_masked_akt
from qikt import bench_qikt
from dkvmn import bench_dkvmn
from deep_irt import bench_deep_irt

datasets1 = ['assist2009', 'algebra2005']
datasets2 = ['duolingo2018_es_en', 'corr_assist2009']
datasets = datasets1 + datasets2

#DKT related
bench_dkt.main(datasets)
bench_mask_label_dkt.main(datasets)
bench_dkt_fuse.main(datasets)

#AKT related
bench_akt.main(datasets)
bench_mask_label_akt.main(datasets)
bench_question_masked_akt.main(datasets)

#DKVMN, DeepIRT, QIKT
bench_qikt.main(datasets)
bench_dkvmn.main(datasets)
bench_deep_irt.main(datasets)