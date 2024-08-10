from kqn import bench_kqn, bench_masked_kqn

datasets1 = ['duolingo2018_es_en', 'corr_assist2009']
datasets2 = ['assist2009', 'algebra2005']
bench_kqn.main(datasets1)
bench_masked_kqn.main(datasets1)
#round 2
print('start round 2...')
bench_kqn.main(datasets2)
bench_masked_kqn.main(datasets2)


