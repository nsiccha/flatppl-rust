parameters {
  cholesky_factor_corr[3] L;
}
model {
  L ~ lkj_corr_cholesky(2.0);
}
