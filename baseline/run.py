from dkt import bench_dkt, bench_masked_dkt
from kqn import bench_kqn, bench_masked_kqn
from akt import bench_akt, bench_masked_akt

datasets = ['assist2009', 'corr_assist2009', 'duolingo2008_es_en', 'algebra2005']
bench_dkt.main(datasets)
bench_masked_dkt.main(datasets)
bench_kqn.main(datasets)
bench_masked_kqn.main(datasets)
bench_akt.main(datasets)
bench_masked_akt.main(datasets)



