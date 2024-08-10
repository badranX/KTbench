from dkvmn import bench_dkvmn
from deep_irt import bench_deep_irt

#datasets1 = ['duolingo2018_es_en']
datasets2 = ['assist2009', 'algebra2005']
datasets = datasets2
bench_deep_irt.main(datasets)
#bench_masked_dkt.main(datasets1)
#round two
bench_dkvmn.main(datasets)
#bench_masked_dkt.main(datasets2)