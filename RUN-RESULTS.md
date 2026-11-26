# Run results
- CV (5-fold, train part only, rerun Nov 26): see results/cv_results.json. Best AUC 0.8481 (stacking), best accuracy 80.0% (stacking). The first version ran CV on all rows incl. the test part; same model won, same test numbers.
- Final test: 79.9% accuracy at threshold 0.53 (79.5% at 0.5), AUC 0.836, 1,407 unseen customers.
- Tests: 3/3 pass. Not deployed.
