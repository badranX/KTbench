from dkt import bench_dkt, bench_masked_dkt
from kqn import bench_kqn, bench_masked_kqn
from akt import bench_akt, bench_masked_akt

datasets1 = ['duolingo2018_es_en',  'corr_assist2009']
datasets2 = ['assist2009', 'algebra2005']
bench_akt.main(datasets1)
bench_masked_akt.main(datasets1)
#round 2
bench_akt.main(datasets2)
bench_masked_akt.main(datasets2)



